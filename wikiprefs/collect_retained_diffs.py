import argparse
import csv
import logging
import os
import re
import time
import xml.sax
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from wikiprefs.diffs.diff_creation import SectionsComparator
from wikiprefs.diffs.diff_saver import DIFF_CSV_FIELDNAMES, DiffRow, DiffSaver
from wikiprefs.diffs.filtering import CommentFilter
from wikiprefs.diffs.model import SectionDiff, TextDiff
from wikiprefs.diffs.tokenization import ProblematicTokenizationPatternException, TooLongSectionException
from wikiprefs.markup.markup_processing import (
    WikiBrokenMarkupException,
    WikiDuplicatedSectionException,
    WikiMarkupParser,
    sections_to_dict,
)
from wikiprefs.markup.revisions_chain import RevisionsChainCreator
from wikiprefs.utils.log_utils import setup_logger
from wikiprefs.utils.xml_utils import Page, PageHistoryHandler, Revision

logger = setup_logger(log_level=logging.INFO, filename='collect_retained_diffs.log')


class DiffCollector:
    """Creates and saves diffs for each revision"""

    def __init__(self, csv_file, context_len, latest_revisions_queue: deque[Revision], revisions_chain: set[str]):
        """Initialize Diff collector

        Args:
            csv_file (str): path to csv file
            context_len (int): number of context lines for unified diffs
            latest_revisions_queue: `n` latest revisions of the article, that will be used as a
                                    reference point for good text. `n` versions are passed in case the latest
                                    have broken markup
            revisions_chain: set of revisions to process (allows to skip e.g. reverted revisions)
        """
        self.parser = WikiMarkupParser()
        self.comparator = SectionsComparator()
        self.comments_filter = CommentFilter()
        self.csv_file = csv_file
        self.context_len = context_len
        self.whitespace_re = re.compile(r'\s+')

        self.latest_rev_id = None
        self.latest_rev_sections = None
        self.latest_rev_processed = False
        self._get_latest_revision_text(latest_revisions_queue)
        latest_revisions_queue.clear()

        self.revisions_chain = revisions_chain

        self.previous_revision: Revision | None = None
        self.old_sections_tokenized = None

        self.diffs_count = 0
        self.all_revisions_count = 0
        self.processed_revisions_count = 0

        # write CSV header if needed
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=DIFF_CSV_FIELDNAMES)
                writer.writeheader()

    def _get_latest_revision_text(self, latest_revisions_queue: deque[Revision]):
        revision_sections = None
        for revision in reversed(latest_revisions_queue):
            revision_text = ''.join(revision.text)
            try:
                revision_sections = sections_to_dict(self.parser.parse_wiki_markup(revision_text))
            except (WikiDuplicatedSectionException, WikiBrokenMarkupException) as e:
                logger.warning(f'Invalid structure for latest revision {revision.id}: {e}')
            else:
                logger.info(f'Using revision {revision.id} as latest article revision')
                self.latest_rev_id = revision.id
                break

        if revision_sections is None:
            logger.error('Failed to find any correct latest revision')
            revision = latest_revisions_queue[-1]
            revision_text = ''.join(revision.text)
            revision_sections = sections_to_dict(self.parser.parse_wiki_markup(revision_text, ignore_errors=True))
            self.latest_rev_id = revision.id
            logger.info(f'Using revision {revision.id} as latest article revision')

        self.latest_rev_sections = revision_sections
        for k, v in self.latest_rev_sections.items():
            self.latest_rev_sections[k] = self._normalize_text(v)

    def __call__(self, page: Page, revision: Revision):
        """Process next revision"""
        self.all_revisions_count += 1

        if self.latest_rev_processed:
            # ignore everything after revision that we're using latest revision
            # (due to markup processing problems it might not be the really latest revision)
            return
        elif revision.id == self.latest_rev_id:
            self.latest_rev_processed = True

        if revision.id not in self.revisions_chain:
            logger.debug(f'Revision {revision.id} is not in revisions chain; skipping')
            return

        if self.previous_revision and self._revision_deletes_article(revision):
            logger.warning(
                f'The text in {revision.id} is 10 times smaller than previous revision '
                f'({self.previous_revision.text_size} vs {revision.text_size}). Skipping the revision'
            )
            return

        new_text = ''.join(revision.text)
        try:
            # parse markup of the new revision
            sections_new = self.parser.parse_wiki_markup(new_text)
        except (WikiDuplicatedSectionException, WikiBrokenMarkupException) as e:
            logger.debug(f'Invalid structure for article {page.title} and revision {revision.id}: {e}')
            return

        # keep only sections that are present in the latest revision, since only their diffs will be preserved
        latest_rev_sections = self.latest_rev_sections.keys()
        sections_new = [s for s in sections_new if s[0] in latest_rev_sections]

        try:
            # tokenize text from the new sections
            sections_new_tokenized = self.comparator.tokenize_sections(sections_new)
        except TooLongSectionException as e:
            logger.warning(f'Too long text in page {page.title} section {e.section_title} and rev {revision.id}: {e}')
            return
        except ProblematicTokenizationPatternException as e:
            logger.warning(
                f'Problematic tokenization pattern detected in text '
                f'for page {page.title} and revision {revision.id}: {e}'
            )
            return

        if self.previous_revision is None:
            # the oldest revision, nothing to compare to
            self.previous_revision = revision
            self.old_sections_tokenized = sections_new_tokenized
            return

        # create diffs
        sections_old_tokenized = self.old_sections_tokenized
        diffs = self.comparator.compare_sections(
            sections_old_tokenized, sections_new_tokenized, context_lines=self.context_len
        )
        self._save_diffs(page, revision, diffs)

        # save current revision as the previous revision for the next run
        self.previous_revision = revision
        self.old_sections_tokenized = sections_new_tokenized
        self.processed_revisions_count += 1

    def _revision_deletes_article(self, revision):
        old_text_size = self.previous_revision.text_size
        new_text_size = revision.text_size
        return new_text_size < old_text_size / 10

    def _save_diffs(self, page: Page, revision: Revision, section_diffs: [SectionDiff]) -> None:
        """Iterate over diffs, and save those which edited text is in the latest revision"""
        row = DiffRow(
            page_title=page.title,
            page_id=int(page.id),
            rev_id=int(revision.id),
            prev_rev_id=int(self.previous_revision.id),
            timestamp=revision.timestamp,
            contributor=revision.contributor,
            comment=revision.comment,
        )
        is_fa_improvement = self.comments_filter.is_fa_edit(revision.comment)

        with open(self.csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=DIFF_CSV_FIELDNAMES)

            for section_diff in section_diffs:
                if section_diff.section not in self.latest_rev_sections:
                    # section not present in latest revision, ignore the diff
                    continue

                row.section = section_diff.section
                for segment_diff in section_diff.all_segments_diffs:
                    saver = DiffSaver(writer, row, revision, save_all_diffs=is_fa_improvement)
                    for diff in segment_diff.diffs:
                        if isinstance(diff, TextDiff):
                            is_retained = self._is_retained_change(section_diff.section, diff.v1, diff.v2)
                            saver.next_diff(diff, is_retained)
                        else:  # context text (plain str)
                            saver.next_text(diff)

                    saver.finish()
                    self.diffs_count += saver.diffs_count

    def _is_retained_change(self, section, old_text, new_text):
        """Check if the edit was kept in the latest revision"""
        normalized_latest_section_text = self.latest_rev_sections[section]

        if old_text is None:
            # text added in v2, and never improved
            return False
        elif old_text is not None and new_text is None:
            # something removed, potentially useful (e.g. vandalism/pov removal)
            return True
        elif old_text.endswith(new_text):
            # new_text is the ending of the old_text, so begging of a sentence was removed
            # usually this means a broken new_text as a results
            return False
        else:
            normalized_new_text = self._normalize_text(new_text)
            # return True if change is kept in latest version
            return normalized_new_text in normalized_latest_section_text

    def _normalize_text(self, text):
        return self.whitespace_re.sub(' ', text).strip()


class RevisionsPreProcessor:
    """Finds latest revisions in XML based on revision timestamp; plus extract revision chain

    We collect `n` revisions, as it's possible that the latest revision has broken markup and can't be parsed
    The revision chain is the history of the article with skipped reverted revision
    """

    def __init__(self, max_revisions: int = 5):
        """Initialize latest revision finder"""
        self.latest_revisions_queue: deque[Revision] = deque(maxlen=max_revisions)
        self.latest_revision_timestamp = None

        self.revision_chain_creator = RevisionsChainCreator()
        self.revision_chain_creator.on_page_started()

    def __call__(self, page: Page, revision: Revision):
        """Process next revision"""
        timestamp = datetime.fromisoformat(revision.timestamp)

        if self.latest_revision_timestamp is None or self.latest_revision_timestamp < timestamp:
            self.latest_revisions_queue.append(revision)
            self.latest_revision_timestamp = timestamp

        self.revision_chain_creator.on_revision_processed(revision)


def process_article(src_xml_path: str, dest_csv_path: str, context_len: int) -> None:
    """Process single article and create diffs for each revision

    Args:
        src_xml_path (str): path to source XML file
        dest_csv_path (str): path to destination CSV file
        context_len (int): number of context lines for unified diffs
    """
    if os.path.exists(dest_csv_path):
        logger.info(f'File {dest_csv_path} already exists. Skipping')
        return

    in_progress_csv_path = f'{dest_csv_path}.INP'
    if os.path.exists(in_progress_csv_path):
        logger.warning(f'File {in_progress_csv_path} already existing, removing')
        os.remove(in_progress_csv_path)

    logger.info(f'Parsing {src_xml_path}')
    start_time = time.time()

    try:
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)  # turn off name spaces

        # find text of the latest revision (last entry in the xml)
        # it will serve as a reference point for "good" edits
        rev_preprocessor = RevisionsPreProcessor()
        handler = PageHistoryHandler(rev_preprocessor)
        parser.setContentHandler(handler)
        parser.parse(src_xml_path)
        logger.debug(f'Found latest revision fpr {src_xml_path}: {rev_preprocessor.latest_revisions_queue[-1]}')

        logger.info(f'Revision chain for {src_xml_path}:')
        revisions_chain = rev_preprocessor.revision_chain_creator.on_page_processed()
        revisions_chain = set([rev.id for rev in revisions_chain])

        # process all revisions, and create differences (diffs) between them
        # keep only text edits that make it to the latest revision
        diff_creator = DiffCollector(
            in_progress_csv_path, context_len, rev_preprocessor.latest_revisions_queue, revisions_chain
        )
        handler = PageHistoryHandler(diff_creator)
        parser.setContentHandler(handler)
        parser.parse(src_xml_path)
    except Exception as e:
        logger.error(f'Error processing {src_xml_path}: {e}')
        logger.error(e, exc_info=True)
        if os.path.exists(in_progress_csv_path):
            os.rename(in_progress_csv_path, f'{dest_csv_path}.ERROR')
    else:
        logger.info(f'Processed {src_xml_path}')
        logger.info(f'Processed {diff_creator.processed_revisions_count}/{diff_creator.all_revisions_count} revisions')
        logger.info(f'Found {diff_creator.diffs_count} diffs for {src_xml_path}')
        os.rename(in_progress_csv_path, dest_csv_path)

    elapsed_time = time.time() - start_time
    logger.info(f'Finished parsing {src_xml_path}, took {elapsed_time}')


def get_metahistory_files(src_dir: str) -> [str]:
    """Get list of source XML meta-history files sorted by size"""
    metahistory_files = []
    for f in os.listdir(src_dir):
        if not os.path.isfile(os.path.join(src_dir, f)):
            continue
        s = os.path.splitext(f)
        if s[-1] == '.xml':
            full_path = os.path.join(src_dir, f)
            file_size = os.path.getsize(full_path)
            metahistory_files.append((s[0], file_size))

    # sort the list by file size, to avoid processing one big file in the end
    # (single thread processing single last file unnecessarily prolongs the process)
    metahistory_files.sort(key=lambda x: x[1], reverse=True)
    return [filename for filename, size in metahistory_files]


def collect_retained_diffs(config):
    """Create and save diffs for all articles"""
    start_time = time.time()

    metahistory_files = get_metahistory_files(config.src)
    logger.info(f'There are {len(metahistory_files)} metahistory files')

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {}
        # submit each article history file for processing
        for metahistory_file_name in metahistory_files:
            src_xml_path = os.path.join(config.src, f'{metahistory_file_name}.xml')
            dst_csv_path = os.path.join(config.dst, f'{metahistory_file_name}.csv')
            f = executor.submit(process_article, src_xml_path, dst_csv_path, config.context_len)
            futures[f] = metahistory_file_name

        total = 0
        for future in as_completed(futures.keys()):
            exception = future.exception()
            if exception is not None:
                logger.error(f'Error processing article {futures[future]}: {exception}')
            else:
                total += 1
                if total % 10 == 0:
                    logger.info(f'Processed {total} articles in {time.time() - start_time}')

    elapsed_time = time.time() - start_time
    logger.info(f'Total parsing took {elapsed_time}')


def setup_arg_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src',
        type=str,
        help='source directory with featured articles history files',
        required=True,
    )
    parser.add_argument(
        '--dst',
        type=str,
        help='destination directory for saving extracted diffs',
        required=True,
    )
    parser.add_argument(
        '--context-len', type=int, help='prefix ans suffix sentences to keep', required=False, default=3
    )
    return parser


def main():
    """Main"""
    parser = setup_arg_parser()
    config = parser.parse_args()

    logger.info(f'Config: {config}')
    assert os.path.isdir(config.src)
    assert os.path.isdir(config.dst)

    collect_retained_diffs(config)


if __name__ == '__main__':
    main()
