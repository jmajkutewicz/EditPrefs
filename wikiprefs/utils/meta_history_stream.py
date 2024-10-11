import bz2
import csv
import io
import json
import multiprocessing
import os
import time
import xml.sax
from argparse import ArgumentParser
from collections.abc import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

logger = multiprocessing.get_logger()


def configure_arg_parser(default_processed_bz2_csv: str, parser: ArgumentParser):
    """Configure argument parser for stream processing Wikipedia meta-history archives"""
    parser.add_argument(
        '--dst',
        type=str,
        help='destination directory for saving extracted history',
        required=True,
    )
    parser.add_argument(
        '--dump-url',
        type=str,
        help='Wikipedia dumps url',
        required=False,
        default='https://dumps.wikimedia.org',
    )
    parser.add_argument(
        '--dump',
        type=str,
        help='Wikipedia dump status file',
        required=False,
        default='https://dumps.wikimedia.org/enwiki/20240401/dumpstatus.json',
    )
    parser.add_argument('--chunk-size', type=int, help='Chunk size in MB', required=False, default=1)
    parser.add_argument(
        '--processed-bz2',
        type=str,
        help='path to csv with list of processed bz2',
        required=False,
        default=default_processed_bz2_csv,
    )
    parser.add_argument('--workers', type=int, help='Number of processes to start', required=False, default=3)
    parser.add_argument(
        '--nodes',
        type=int,
        required=False,
        help='Number of computer nodes used for downloading and extracting Wikipedia articles history',
        default=1,
    )
    parser.add_argument('--node-id', type=int, help='Node id', required=False, default=0)


def get_pages_metahistory_archives_list(dump_status_url: str) -> [str]:
    """Get list of all meta-history .bz2 files"""
    logger.info(f'Getting list of all meta-history .bz2 files from : {dump_status_url}')
    response = requests.get(dump_status_url)
    if response.status_code != 200:
        logger.error(f'Failed to download featured articles page. Status code: {response.status_code}')
        logger.error(f'{response.text}')
        exit(1)

    dump_status = json.loads(response.text)
    metahistory_bz2 = dump_status['jobs']['metahistorybz2dump']['files']
    metahistory_bz2_slugs = [bz2['url'] for bz2 in metahistory_bz2.values()]

    logger.info(f'There are {len(metahistory_bz2_slugs)} bz2 files in the dump')
    return metahistory_bz2_slugs


def get_node_split(metahistory_bz2_slugs: [str], nodes_count: int, node_id: int) -> [str]:
    """Decide which .bz2 files should be processed on this node (in case more than 1 node is processing the dump)"""
    if nodes_count == 1:
        return metahistory_bz2_slugs

    per_node = len(metahistory_bz2_slugs) // nodes_count
    remainder = len(metahistory_bz2_slugs) % nodes_count

    if node_id < remainder:
        # Nodes before the remainder point get 'per_node + 1' elements.
        start_index = node_id * (per_node + 1)
        end_index = start_index + (per_node + 1)
    else:
        # Nodes after the remainder point just get 'per_node' elements,
        start_index = remainder * (per_node + 1) + (node_id - remainder) * per_node
        end_index = start_index + per_node
    logger.info(f'Node split: {start_index} - {end_index}')

    node_slugs = metahistory_bz2_slugs[start_index:end_index]
    logger.info(f'There are {len(node_slugs)} bz2 files to process on this node')
    logger.info(f'\tFirst archive: {node_slugs[0]}')
    logger.info(f'\tLast archive: {node_slugs[-1]}')
    return node_slugs


def filter_processed_archives(metahistory_bz2_slugs: [str], processed_bz2_csv: str) -> [str]:
    """Filter out .bz2 files that were already processed"""
    if not os.path.exists(processed_bz2_csv):
        return metahistory_bz2_slugs

    with open(processed_bz2_csv, encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        processed_archives = set()
        for row in reader:
            processed_archives.add(row[0])

    filtered_slugs = [s for s in metahistory_bz2_slugs if s not in processed_archives]
    logger.info(f'There are {len(filtered_slugs)} bz2 files still to process')
    return filtered_slugs


def save_processed_archive(processed_bz2_csv: str, slug: str, total_pages: int, saved_pages: int) -> None:
    """Save information that this .bz2 file was processed"""
    if not os.path.exists(processed_bz2_csv):
        with open(processed_bz2_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ['slug', 'total_pages', 'saved_pages']
            writer.writerow(header)

    with open(processed_bz2_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow((slug, total_pages, saved_pages))


def get_requests_session(max_retries=10, backoff_factor=2):
    """Setup HTTP session"""
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def download_with_retry(
    url: str, chunk_size_mb: int, stream_consumer: Callable[[bytes], None], max_retries=10, retry_delay=60
):
    """Download .bz2 file. Handle errors, rate limiting, etc."""
    bytes_downloaded = 0
    retry = 0
    session = get_requests_session(max_retries=3, backoff_factor=1)

    while True:
        try:
            headers = {'Range': f'bytes={bytes_downloaded}-'}
            with session.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()  # Ensure the request was successful
                for chunk in r.iter_content(chunk_size=chunk_size_mb * 1024 * 1024):  # Read in 1MB chunks
                    if chunk:  # Filter out keep-alive new chunks
                        stream_consumer(chunk)
                        bytes_downloaded += len(chunk)

        except requests.exceptions.RequestException as e:
            if isinstance(
                e,
                requests.exceptions.ConnectionError
                | requests.exceptions.ConnectTimeout
                | requests.exceptions.ReadTimeout
                | requests.exceptions.Timeout
                | requests.exceptions.ChunkedEncodingError,
            ):
                retry += 1
                logger.warning(f'Network error while processing {url} - retrying ({retry}). Exception: {e}')
                if retry == max_retries:
                    logger.error(f'Reached maximum number of retries ({retry})')
                    raise e
                else:
                    time.sleep(retry_delay)
                    continue
            else:
                logger.error(f'Exception while processing {url}: {e}')
                raise e
        else:
            logger.info(f'Finished processing {url} successfully')
            break


class StreamConsumer:
    """HTTP stream consumer, that processes meta-history bz2 files on the fly

    This class processes the .bz2 HTTP stream by chunks, extract the accumulated bz2 to XML
    and passes the XML chunks to MetaHistoryXmlHandler.
    This way we don't need to save neither the .bz2 files nor the whole extracted XML to disk.
    We only save the selected articles meta-history.
    """

    def __init__(self, content_handler: xml.sax.ContentHandler):
        """Initialize the Stream Consumer"""
        self._decompressor = bz2.BZ2Decompressor()
        self._buffer = io.BytesIO()
        self._parser = xml.sax.make_parser()

        self._handler = content_handler
        self._parser.setContentHandler(content_handler)

    def __call__(self, chunk: bytes):
        """Process the next chunk of the stream"""
        data = self._decompressor.decompress(chunk)
        if data:
            self._buffer.write(data)

            try:
                self._buffer.seek(0)  # Go to the start of the buffer to read from it
                self._parser.feed(self._buffer.read())  # Parse the data
                self._reset_buffer()  # Reset buffer after successful parsing
            except xml.sax.SAXException:
                # Handle the case where the buffer ends with incomplete XML
                # (just continue accumulating data into the buffer)
                self._buffer.seek(0, io.SEEK_END)

    def finalize(self):
        """Finish stream processing"""
        # Handle any remaining data after all chunks have been processed.
        try:
            self._buffer.seek(0)
            self._parser.feed(self._buffer.read())
        except xml.sax.SAXException as e:
            logger.error(f'Error during final parsing: {e}')

    def get_processed_pages(self) -> (int, int):
        """Get number of pages processed and saved"""
        return self._handler.get_processed_pages()

    def _reset_buffer(self):
        self._buffer.seek(0)
        self._buffer.truncate()  # Clear the content
