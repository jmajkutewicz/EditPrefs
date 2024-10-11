import csv
import logging
from argparse import ArgumentParser
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from wikiprefs.utils.log_utils import setup_logger

logger = setup_logger(log_level=logging.INFO, filename='download_featured_articles_list.log')

FEATURED_ARTICLES_PAGE_URL = 'https://en.wikipedia.org/wiki/Wikipedia:Featured_articles'
URL_PREFIX = '/wiki/'


def download_featured_articles_list(csv_file_path: str) -> None:
    """Extract featured articles list from Wikipedia"""
    logger.info('Downloading featured articles list')

    response = requests.get(FEATURED_ARTICLES_PAGE_URL)
    if response.status_code != 200:
        logger.error(f'Failed to featured articles page. Status code: {response.status_code}')
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    featured_articles = []

    # Find all the relevant spans and extract the data
    for span in soup.find_all('span', class_='featured_article_metadata has_been_on_main_page'):
        a_tag = span.find('a')
        if a_tag:
            article_title = a_tag['title']
            article_slug = a_tag['href']

            assert article_slug.startswith(URL_PREFIX)
            article_slug = article_slug[len(URL_PREFIX) :]

            featured_articles.append((article_title, article_slug))

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['article_title', 'article_slug']
        writer.writerow(header)
        for article in featured_articles:
            writer.writerow(article)
    logger.info('Successfully downloaded featured articles list')


def main():
    """Main"""
    parser = ArgumentParser(
        prog='Wikipedia featured articles list downloader', description='Download featured articles list from Wikipedia'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='result csv file path',
        required=False,
        default=Path(__file__).parent / 'tmp/featured_articles.csv',
    )
    config = parser.parse_args()

    download_featured_articles_list(config.csv)


if __name__ == '__main__':
    main()
