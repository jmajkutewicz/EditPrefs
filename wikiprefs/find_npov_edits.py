from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import re
import time
import urllib.request
import xml.sax
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import requests

from wikiprefs.diffs.filtering import CommentFilter
from wikiprefs.markup.revisions_chain import RevisionNode, RevisionsChainCreator
from wikiprefs.utils.log_utils import setup_logger
from wikiprefs.utils.npov import NPOV_FIELDNAMES, NpovEdit
from wikiprefs.utils.xml_utils import MetaHistoryXmlHandler, Page, PageHistoryConsumerInterface, Revision

logger = setup_logger(log_level=logging.INFO, filename='find_npov_edits.log')

WIKI_BASE_URL = 'https://dumps.wikimedia.org'


class PageHistoryConsumer(PageHistoryConsumerInterface):
    """Page history consumer, that collect chain of non-reverted revisions and then find NPOV edits"""

    def __init__(self, csv_file):
        self.revision_chain_creator = RevisionsChainCreator()
        self.comment_filter = CommentFilter()

        self.title_re = re.compile(r'^.*?(?:talk|user|wikipedia|template|help|file|draft|category):', re.IGNORECASE)
        self.curr_page: Page | None = None
        self.ignore_page = True

        self.csv_file = csv_file
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        # write CSV header
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=NPOV_FIELDNAMES)
            writer.writeheader()

    def on_page_started(self) -> None:
        self.revision_chain_creator.on_page_started()
        self.curr_page = None
        self.ignore_page = True

    def on_page_processed(self) -> None:
        if self.ignore_page:
            return

        revisions = self.revision_chain_creator.on_page_processed()
        self._save_npov_edits(revisions)

    def on_revision_processed(self, page: Page, revision: Revision) -> None:
        if self.curr_page is None:
            self.curr_page = page
            self.ignore_page = self.title_re.match(page.title)
            if self.ignore_page:
                logger.info(f'Ignoring page {page.title}')
        if self.ignore_page:
            return

        self.revision_chain_creator.on_revision_processed(revision)

    def _save_npov_edits(self, revisions: list[RevisionNode]) -> None:
        if self.curr_page is None:
            logger.error('Page is missing for saving NPOV edits')
            return

        with open(self.csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=NPOV_FIELDNAMES)

            for revision in revisions:
                comment = revision.comment
                if not self.comment_filter.is_npov_edit(comment):
                    continue

                logger.info(f'Found NPOV edit for page {self.curr_page.title}: [{revision.timestamp}] {comment} ')
                parent_id = revision.parent.id if revision.parent is not None else ''

                npov_edit = NpovEdit(
                    self.curr_page.id,
                    self.curr_page.title,
                    revision.id,
                    parent_id,
                    comment,
                )
                writer.writerow(asdict(npov_edit))


def process_metahistory_stub(metahistory_stub_name, metahistory_stub_filepath, csv_file) -> None:
    """Process single meta-history-stub archive: extract NPOV edits metadata"""
    logger.info(f'Starting processing {metahistory_stub_name}')
    start_time = time.time()

    try:
        consumer = PageHistoryConsumer(csv_file)
        process_metahistory_stubs_xml(consumer, metahistory_stub_filepath)
    except Exception as e:
        logger.error(f'Error processing {metahistory_stub_name}: {e}')
        logger.error(e, exc_info=True)
        if os.path.exists(csv_file):
            os.rename(csv_file, f'{csv_file}.ERROR')

    elapsed_time = time.time() - start_time
    logger.info(f'{os.getpid()}\tParsing {metahistory_stub_name} took {elapsed_time}s')


def process_metahistory_stubs_xml(consumer: PageHistoryConsumer, metahistory_stub_filepath: str) -> None:
    """Process meta-history-stub compressed XML"""
    handler = MetaHistoryXmlHandler(consumer)
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)  # turn off name spaces
    parser.setContentHandler(handler)
    with gzip.open(metahistory_stub_filepath, 'rt', encoding='utf-8') as f:
        parser.parse(f)


def download_stub_archive(stub_slug: str, stub_file: str):
    """Download the meta-history stub archive to disk"""
    logger.info(f'Downloading {stub_slug} to {stub_file}')
    url = f'{WIKI_BASE_URL}/{stub_slug}'
    urllib.request.urlretrieve(url, stub_file)


def get_metahistory_stub(dump_status_url: str, archives_dir: str) -> [tuple[str, str]]:
    """Get list of all meta-history .bz2 files"""
    response = requests.get(dump_status_url)
    if response.status_code != 200:
        logger.error(f'Failed to download featured articles page. Status code: {response.status_code}')
        exit(1)

    dump_status = json.loads(response.text)
    metahistory_stubs = dump_status['jobs']['xmlstubsdump']['files']
    metahistory_stubs = [stub['url'] for stub in metahistory_stubs.values()]

    stub_re = re.compile(r'/enwiki/\d+/(enwiki-\d+-stub-meta-history\d+).xml.gz')
    stubs = []
    futures = {}
    with ProcessPoolExecutor(max_workers=3) as executor:
        for stub_url in metahistory_stubs:
            match = stub_re.match(stub_url)
            if not match:
                continue

            stub_name = match.group(1)
            stub_file = os.path.join(archives_dir, f'{stub_name}.xml.gz')
            stubs.append((stub_name, stub_file))

            if os.path.exists(stub_file):
                logger.info(f'Skipping {stub_name}, already downloaded')
                continue

            f = executor.submit(download_stub_archive, stub_url, stub_file)
            futures[f] = stub_name

        for f in as_completed(futures):
            exception = f.exception()
            if exception is not None:
                logger.error(f'Error downloading stub archive {futures[f]}: {exception}')
            else:
                logger.info(f'Downloaded stub {futures[f]}')

    logger.info(f'There are {len(stubs)} meta-history stubs archives')
    stubs.sort(key=lambda s: s[0])
    return stubs


def filter_processed_stubs(metahistory_stubs: list[tuple[str, str]], tmp_dir: Path) -> list[tuple[str, str]]:
    """Filter out already processed meta-history stub archives"""
    filtered_stubs = []
    for stub in metahistory_stubs:
        csv_file = tmp_dir / f'{stub[0]}.csv'
        if not os.path.exists(csv_file):
            filtered_stubs.append(stub)
    return filtered_stubs


def main():
    """Main"""
    parser = ArgumentParser(
        prog='Wikipedia NPOV edits ids extractor',
        description='Extracts NPOV edits ids from Wikipedia dump',
    )
    parser.add_argument(
        '--dump',
        type=str,
        help='Wikipedia dump status file',
        required=False,
        default='https://dumps.wikimedia.org/enwiki/20240401/dumpstatus.json',
    )
    parser.add_argument(
        '--stub-files-dir',
        type=str,
        help='directory for meta-history stub archives',
        required=True,
    )
    parser.add_argument(
        '--dst',
        type=str,
        help='destination directory for NPOV edits list',
        default=Path(__file__).parent / 'tmp',
        required=False,
    )
    config = parser.parse_args()
    logger.info(f'Config: {config}')
    assert os.path.exists(config.stub_files_dir)
    assert os.path.exists(config.dst)

    tmp_dir = Path(__file__).parent / 'tmp' / 'npov_edits'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    metahistory_stubs = get_metahistory_stub(config.dump, config.stub_files_dir)
    # metahistory_stubs = metahistory_stubs[:8]
    metahistory_stubs = filter_processed_stubs(metahistory_stubs, tmp_dir)
    logger.info(f'There are {len(metahistory_stubs)} meta-history stubs to process')

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {}
        for metahistory_stub in metahistory_stubs:
            name = metahistory_stub[0]
            archive = metahistory_stub[1]
            csv_file = tmp_dir / f'{name}.csv'
            f = executor.submit(process_metahistory_stub, metahistory_stub, archive, csv_file)
            futures[f] = name

        for future in as_completed(futures):
            exception = future.exception()
            if exception is not None:
                logger.error(f'Error processing meta-history stub {futures[future]}: {exception}')
            else:
                logger.info(f'Processed meta-history stub {futures[future]}')

    tmp_csv_files = os.listdir(tmp_dir)
    if any(s.endswith('.ERROR') for s in tmp_csv_files):
        logger.error('Failed to process all meta-history stub archives correctly')
        exit(1)

    dst_csv_file = os.path.join(config.dst, 'npov_edits.csv')
    with open(dst_csv_file, 'w', newline='', encoding='utf-8') as dst_csv:
        csv.DictWriter(dst_csv, fieldnames=NPOV_FIELDNAMES).writeheader()

        for tmp_csv_file in tmp_csv_files:
            with open(tmp_dir / tmp_csv_file, newline='', encoding='utf-8') as src_csv:
                src_csv.readline()  # skip header
                for line in src_csv:
                    dst_csv.write(line)


if __name__ == '__main__':
    main()
