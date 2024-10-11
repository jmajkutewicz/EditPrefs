import csv
import logging
import os
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from wikiprefs.diffs.diff_creation import SectionsComparator
from wikiprefs.diffs.diff_saver import DIFF_CSV_FIELDNAMES, DiffRow, DiffSaver
from wikiprefs.diffs.model import SectionDiff, TextDiff
from wikiprefs.diffs.tokenization import ProblematicTokenizationPatternException, TooLongSectionException
from wikiprefs.markup.markup_processing import (
    WikiBrokenMarkupException,
    WikiDuplicatedSectionException,
    WikiMarkupParser,
)
from wikiprefs.utils.log_utils import setup_logger
from wikiprefs.utils.meta_history_stream import (
    StreamConsumer,
    configure_arg_parser,
    download_with_retry,
    filter_processed_archives,
    get_node_split,
    get_pages_metahistory_archives_list,
)
from wikiprefs.utils.npov import NPOV_FIELDNAMES, NpovEdit
from wikiprefs.utils.xml_utils import MetaHistoryXmlHandler, Page, PageHistoryConsumerInterface, Revision

logger = setup_logger(
    log_level=logging.INFO, filename=os.getenv('LOG_FILE', default='collect_npov_diffs.log'), use_stdout_handler=False
)


class PageHistoryConsumer(PageHistoryConsumerInterface):
    """Page history consumer, that collect chain of non-reverted revisions and then find NPOV edits"""

    def __init__(
        self, pages_to_extract: set[str], page2edits: dict[str, list[NpovEdit]], csv_file: str, context_len: int = 3
    ):
        self.pages_to_extract = pages_to_extract
        self.page2edits = page2edits
        self.csv_file = csv_file
        self.context_len = context_len

        self.parser = WikiMarkupParser()
        self.comparator = SectionsComparator()

        self.current_page: Page | None = None
        self.skip_page: bool = True

        self.current_edits: list[NpovEdit] | None = None
        self.current_edit: NpovEdit | None = None
        self.current_i: int = -1
        self.current_parent_rev: Revision | None = None

        self.revision_diffs_total = 0
        self.revision_diffs_processed = 0
        self.diffs_count = 0

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=DIFF_CSV_FIELDNAMES)
                writer.writeheader()

    def on_page_started(self) -> None:
        self.current_page = None
        self.skip_page = True

        self.current_edits = None
        self.current_edit = None
        self.current_i = -1
        self.current_parent_rev = None

    def on_page_processed(self) -> None:
        if self.current_page and self.current_edits:
            logger.info(f'Finished processing page {self.current_page.title}')

        if self.current_edits and self.current_i < len(self.current_edits):
            logger.error(f'Failed to process all edits for page {self.current_page.title}')
            logger.error(f'Processed edits: {self.current_i} / {len(self.current_edits)}')

    def on_revision_processed(self, page: Page, revision: Revision) -> None:
        if self.current_page is None:
            self.current_page = page
            self.skip_page = page.id not in self.page2edits
            if self.skip_page:
                logger.debug(f'Ignoring page {page.title}')
            else:
                logger.info(f'Starting NPOV diffs extraction for page {self.current_page.title}')
                edits = self.page2edits[page.id]
                self.current_edits = edits

                self.current_i = 0
                self.current_edit = self.current_edits[self.current_i]

        if self.skip_page:
            return

        rev_id = revision.id
        if rev_id == self.current_edit.parent_rev_id:
            self.current_parent_rev = revision
        elif rev_id == self.current_edit.rev_id:
            if self.current_parent_rev is None:
                logger.warning(f'Revision {rev_id} (page {page.title}) has no parent')
            else:
                self._save_diffs(self.current_page, self.current_parent_rev, revision)

            self.current_i += 1
            if self.current_i < len(self.current_edits):
                self.current_edit = self.current_edits[self.current_i]
                self.current_parent_rev = revision if rev_id == self.current_edit.parent_rev_id else None
            else:
                self.current_edit = None
                self.current_parent_rev = None

                self.skip_page = True  # extracted all NPOV edits for current page

    def _save_diffs(self, page: Page, parent_rev: Revision, rev: Revision) -> None:
        try:
            self.revision_diffs_total += 1

            if self._revision_deletes_article(rev, parent_rev):
                logger.warning(
                    f'The text in {rev.id} is 10 times smaller than previous revision '
                    f'({parent_rev.text_size} vs {rev.text_size}). Skipping the revision'
                )
                return

            # parse markup
            try:
                text_old = ''.join(parent_rev.text)
                sections_old = self.parser.parse_wiki_markup(text_old)

                text_new = ''.join(rev.text)
                sections_new = self.parser.parse_wiki_markup(text_new)
            except (WikiDuplicatedSectionException, WikiBrokenMarkupException) as e:
                logger.debug(f'Invalid structure for article {page.title} and revision {rev.id}: {e}')
                return

            # tokenize text
            try:
                sections_old_tokenized = self.comparator.tokenize_sections(sections_old)
                sections_new_tokenized = self.comparator.tokenize_sections(sections_new)
            except (TooLongSectionException, ProblematicTokenizationPatternException) as e:
                logger.debug(f'Failed to tokenize text for page {page.title} and revision {rev.id}: {e}')
                return

            # create diffs
            section_diffs: [SectionDiff] = self.comparator.compare_sections(
                sections_old_tokenized, sections_new_tokenized, context_lines=self.context_len
            )

            # save diffs
            base_row = DiffRow(
                page_title=page.title,
                page_id=int(page.id),
                rev_id=int(rev.id),
                prev_rev_id=int(parent_rev.id),
                timestamp=rev.timestamp,
                contributor=rev.contributor,
                comment=rev.comment,
            )
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=DIFF_CSV_FIELDNAMES)

                for section_diff in section_diffs:
                    base_row.section = section_diff.section
                    for segment_diff in section_diff.all_segments_diffs:
                        # we try to save all diffs while collecting NPOV edits
                        # this will later allow us to filter out revisions that edited more than 1 paragraph
                        # and ensure we keep only actual NPOV edits and not unrelated edits made in the same revision
                        save_all_diffs = True
                        saver = DiffSaver(writer, base_row, rev, save_all_diffs)
                        for diff in segment_diff.diffs:
                            if isinstance(diff, TextDiff):
                                saver.next_diff(diff, True)
                            else:  # context text (plain str)
                                saver.next_text(diff)

                        saver.finish()
                        self.diffs_count += saver.diffs_count

            self.revision_diffs_processed += 1
        except Exception as e:
            logger.error(f'Failed to process revision {rev.id} (page {page.title}): {e}')
            logger.error(e, exc_info=True)

    def _revision_deletes_article(self, rev: Revision, parent_rev: Revision) -> bool:
        old_text_size = parent_rev.text_size
        new_text_size = rev.text_size
        return new_text_size < old_text_size / 10


def process_metahistory_archive(
    slug: str,
    pages_to_extract: set[str],
    page2edits: dict[str, list[NpovEdit]],
    dst_dir: str,
    chunk_size: int,
    context_len: int,
    wiki_dump_url: str,
) -> tuple[str, int] | None:
    """Download and process .bz2 file in a streaming way"""
    logger.info(f'Starting processing {slug}')
    start_time = time.time()

    csv_file = os.path.join(dst_dir, f'{os.path.basename(slug)}.csv')
    consumer = PageHistoryConsumer(pages_to_extract, page2edits, csv_file, context_len)
    try:
        handler = MetaHistoryXmlHandler(consumer)
        stream_consumer = StreamConsumer(handler)

        url = f'{wiki_dump_url}/{slug}'
        download_with_retry(url, chunk_size, stream_consumer)
        stream_consumer.finalize()
    except Exception as e:
        logger.error(f'Error processing {slug}: {e}')
        logger.error(e, exc_info=True)
        if os.path.exists(csv_file):
            os.rename(csv_file, f'{csv_file}.ERROR')
        return None

    elapsed_time = time.time() - start_time
    logger.info(f'\t\tParsing {slug} took {elapsed_time}s')
    logger.info(
        f'\t\tSuccessfully processed {consumer.revision_diffs_processed} / {consumer.revision_diffs_total} npov diffs'
    )
    return slug, consumer.diffs_count


def get_npov_edits_to_extract(csv_file: str) -> tuple[set[str], dict[str, list[NpovEdit]]]:
    """Get list of pages and edits that should be extracted from .bz2 meta-history files"""
    with open(csv_file) as file:
        reader = csv.DictReader(file, fieldnames=NPOV_FIELDNAMES)

        pages = set()
        page2edits = {}
        npov_edits_count = 0
        for row in reader:
            npov_edit = NpovEdit(
                row['page_id'],
                row['page_title'],
                row['rev_id'],
                row['parent_rev_id'],
                row['comment'],
            )
            page = npov_edit.page_id
            pages.add(page)

            if page not in page2edits:
                page2edits[page] = []
            page2edits[page].append(npov_edit)

            npov_edits_count += 1

    logger.info(f'There are {len(pages)} pages to extract')
    logger.info(f'There are {npov_edits_count} npov edits to extract')
    return pages, page2edits


def save_processed_archive(processed_bz2_csv: str, slug: str, saved_diffs: int) -> None:
    """Save information that this .bz2 file was processed"""
    if not os.path.exists(processed_bz2_csv):
        with open(processed_bz2_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ['slug', 'saved_diffs']
            writer.writerow(header)

    with open(processed_bz2_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow((slug, saved_diffs))


def main():
    """Main"""
    parser = ArgumentParser(
        prog='Wikipedia NPOV edits extractor',
        description='Extracts selected NPOV edits from Wikipedia dump',
    )
    parser.add_argument(
        '--npov-edits',
        type=str,
        help='path to csv with list of npov edits to extract',
        required=False,
        default=Path(__file__).parent / 'tmp/npov_edits.csv',
    )
    parser.add_argument(
        '--context-len', type=int, help='prefix ans suffix sentences to keep', required=False, default=3
    )
    default_processed_bz2_csv = str(Path(__file__).parent / 'tmp/processed_bz2__npov.csv')
    configure_arg_parser(default_processed_bz2_csv, parser)
    config = parser.parse_args()
    logger.info(f'Config: {config}')
    assert os.path.exists(config.dst)

    pages, page2edits = get_npov_edits_to_extract(config.npov_edits)

    metahistory_bz2_slugs = get_pages_metahistory_archives_list(config.dump)
    metahistory_bz2_slugs = get_node_split(metahistory_bz2_slugs, config.nodes, config.node_id)
    metahistory_bz2_slugs = filter_processed_archives(metahistory_bz2_slugs, config.processed_bz2)
    # metahistory_bz2_slugs = metahistory_bz2_slugs[:3]

    # 3 processes, since that's the maximum concurrent connections allowed by Wikipedia
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {}
        for metahistory_bz2_slug in metahistory_bz2_slugs:
            f = executor.submit(
                process_metahistory_archive,
                metahistory_bz2_slug,
                pages,
                page2edits,
                config.dst,
                config.chunk_size,
                config.context_len,
                config.dump_url,
            )
            futures[f] = metahistory_bz2_slug

        for future in as_completed(futures):
            exception = future.exception()
            if exception is not None:
                logger.error(f'Error processing meta-history bz2 archive {futures[future]}: {exception}')
                logger.error(exception, exc_info=True)
            else:
                result = future.result()
                if result is None:
                    logger.error(f'Failed to proces meta-history bz2 archive {futures[future]}')
                else:
                    save_processed_archive(config.processed_bz2, *result)
                    logger.info(f'Processed meta-history bz2 archive {futures[future]}: {result}')


if __name__ == '__main__':
    main()
