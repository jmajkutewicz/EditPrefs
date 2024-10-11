import csv
import logging
import os
import time
import xml.sax
from argparse import ArgumentParser
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

from wikiprefs.utils.log_utils import setup_logger
from wikiprefs.utils.meta_history_stream import (
    StreamConsumer,
    configure_arg_parser,
    download_with_retry,
    filter_processed_archives,
    get_node_split,
    get_pages_metahistory_archives_list,
    save_processed_archive,
)
from wikiprefs.utils.xml_utils import WIKI_XML

logger = setup_logger(log_level=logging.DEBUG, filename='download_articles_history.log')

WIKI_BASE_URL = 'https://dumps.wikimedia.org'


class PageHandler:
    """XML page handler, that decides if page should be discarded or saved (then saves the page in xml file)"""

    class PageStatus(Enum):
        """Page keep/discard status"""

        KEEP = 1
        DISCARD = 2
        UNKNOWN = 3

    def __init__(self, articles_to_extract, dst_dir):
        self._articles_to_extract = articles_to_extract
        self._dst_dir = dst_dir

        self._title = ''
        self._buffer = []
        self._out = None
        self._status = PageHandler.PageStatus.UNKNOWN

    def on_title_loaded(self) -> None:
        self._title = self._title.strip()
        if self._title not in self._articles_to_extract:
            self._status = PageHandler.PageStatus.DISCARD
            logger.debug(f'Discarding article: {self._title}')
        else:
            self._status = PageHandler.PageStatus.KEEP
            logger.info(f'Saving article: {self._title}')

            filename = self._clear_filename(self._title)
            dst_file = os.path.join(self._dst_dir, f'{filename}.xml')
            self._out = open(dst_file, 'w', encoding='utf-8')
            buffer_str = ''.join(self._buffer)
            self._write_to_file(buffer_str)

        self._buffer.clear()

    def on_page_finished(self) -> bool:
        if self._status == PageHandler.PageStatus.KEEP:
            self._out.close()
        return self._status == PageHandler.PageStatus.KEEP

    def update_content(self, content) -> None:
        if self._status == PageHandler.PageStatus.DISCARD:
            pass
        elif self._status == PageHandler.PageStatus.UNKNOWN:
            content = self.escape(content)
            self._buffer.append(content)
        elif self._status == PageHandler.PageStatus.KEEP:
            content = self.escape(content)
            self._write_to_file(content)

    def update_element(self, element) -> None:
        if self._status == PageHandler.PageStatus.DISCARD:
            pass
        elif self._status == PageHandler.PageStatus.UNKNOWN:
            self._buffer.append(element)
        elif self._status == PageHandler.PageStatus.KEEP:
            self._write_to_file(element)

    def update_title(self, content) -> None:
        self._title += self.escape(content)

    def _write_to_file(self, content) -> None:
        self._out.write(content)

    @staticmethod
    def _clear_filename(name: str) -> str:
        unsafe_chars = ['/', '\\', '<', '>', ':', '"', '|', '?', '*']
        for ch in unsafe_chars:
            if ch in name:
                name = name.replace(ch, '_')
        return name

    @staticmethod
    def escape(str_xml: str):
        str_xml = str_xml.replace('&', '&amp;')
        str_xml = str_xml.replace('<', '&lt;')
        str_xml = str_xml.replace('>', '&gt;')
        str_xml = str_xml.replace('"', '&quot;')
        str_xml = str_xml.replace("'", '&apos;')
        return str_xml


class MetaHistoryXmlHandler(xml.sax.ContentHandler):
    """XML content handler, can process meta-history XML dump"""

    def __init__(self, page_handler_factory: Callable[[], PageHandler]):
        super().__init__()
        self.page_handler_factory = page_handler_factory

        self._in_page = False
        self._in_title = False
        self._page_handler = None

        self._processed_pages = 0
        self._saved_pages = 0

    def startElement(self, name, attrs):
        if name == WIKI_XML.PAGE:
            self._in_page = True
            self._page_handler = self.page_handler_factory()

            attributes_str = ' '.join(f'{attr}="{attrs.getValue(attr)}"' for attr in attrs.getNames())
            self._page_handler.update_element(f'<{name} {attributes_str}>')
        elif self._in_page:
            if name == WIKI_XML.TITLE:
                self._in_title = True

            attributes_str = ' '.join(f'{attr}="{attrs.getValue(attr)}"' for attr in attrs.getNames())
            self._page_handler.update_element(f'<{name} {attributes_str}>')

    def characters(self, content):
        if self._in_page:
            self._page_handler.update_content(content)
            if self._in_title:
                self._page_handler.update_title(content)

    def endElement(self, name):
        if self._in_page:
            self._page_handler.update_element(f'</{name}>')

        if name == WIKI_XML.PAGE:
            is_page_saved = self._page_handler.on_page_finished()
            if is_page_saved:
                self._saved_pages += 1

            self._processed_pages += 1
            if self._processed_pages % 10 == 0:
                logger.info(f'Processed {self._processed_pages} pages')

            self._page_handler = None
            self._in_page = False
        elif name == WIKI_XML.TITLE:
            self._page_handler.on_title_loaded()
            self._in_title = False

    def get_processed_pages(self) -> (int, int):
        return self._processed_pages, self._saved_pages


def get_articles_list(csv_file: str) -> set[str]:
    """Get list of articles titles that should be extracted from .bz2 meta-history files"""
    with open(csv_file) as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        articles = set()
        for row in reader:
            articles.add(row[0])

    logger.info(f'There are {len(articles)} articles to extract')
    return articles


def process_metahistory_archive(
    slug: str, articles_to_extract: set[str], dst_dir: str, chunk_size: int
) -> (str, int, int):
    """Download and process .bz2 file in a streaming way"""
    logger.info(f'Starting processing {slug}')
    start_time = time.time()
    url = f'{WIKI_BASE_URL}/{slug}'

    handler = MetaHistoryXmlHandler(lambda: PageHandler(articles_to_extract, dst_dir))
    stream_consumer = StreamConsumer(handler)
    download_with_retry(url, chunk_size, stream_consumer)
    stream_consumer.finalize()

    elapsed_time = time.time() - start_time
    logger.info(f'{os.getpid()}\tParsing {slug} took {elapsed_time}s')

    return (slug, *stream_consumer.get_processed_pages())


def main():
    """Main"""
    parser = ArgumentParser(
        prog='Wikipedia articles history extractor',
        description='Extracts selected articles history from Wikipedia dump',
    )
    parser.add_argument(
        '--articles',
        type=str,
        help='path to csv with articles to download',
        required=False,
        default=Path(__file__).parent / 'tmp/featured_articles.csv',
    )
    default_processed_bz2_csv = str(Path(__file__).parent / 'tmp/processed_bz2__fa.csv')
    configure_arg_parser(default_processed_bz2_csv, parser)
    config = parser.parse_args()
    logger.info(f'Config: {config}')
    assert os.path.exists(config.dst)

    metahistory_bz2_slugs = get_pages_metahistory_archives_list(config.dump)
    metahistory_bz2_slugs = get_node_split(metahistory_bz2_slugs, config.nodes, config.node_id)
    metahistory_bz2_slugs = filter_processed_archives(metahistory_bz2_slugs, config.processed_bz2)
    # metahistory_bz2_slugs = [metahistory_bz2_slugs[0]]
    articles = get_articles_list(config.articles)

    # 3 processes, since that's the maximum concurrent connections allowed by Wikipedia
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {}
        for bz2_archive in metahistory_bz2_slugs:
            f = executor.submit(process_metahistory_archive, bz2_archive, articles, config.dst, config.chunk_size)
            futures[f] = bz2_archive

        for future in as_completed(futures):
            exception = future.exception()
            if exception is not None:
                logger.error(f'Error processing bz2 archive {futures[future]}: {exception}')
            else:
                result = future.result()
                save_processed_archive(config.processed_bz2, *result)
                logger.info(f'Processed bz2 archive {futures[future]}: {result}')


if __name__ == '__main__':
    main()
