import asyncio
import html
import logging
import multiprocessing
import os
import re
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import mwparserfromhell as mwp
import yaml

logger = multiprocessing.get_logger()


class TemplatesCache:
    """Cache for storing rendered templates"""

    def __init__(self, templates_filename):
        """Initialize templates cache

        Args:
            templates_filename (str): name of the file for storing templates
        """
        self.templates_file = os.path.join(Path(__file__).parent.parent, 'resources', templates_filename)
        self.templates = self._load_templates()
        self.lock = asyncio.Lock()

    def get_template(self, template) -> str | None:
        """Returns rendered template if it exists"""
        return self.templates.get(template, None)

    async def add_template(self, template: str, rendered_template: str):
        """Adds rendered template to templates cache and save it to file for future usage"""
        async with self.lock:
            self.templates[template] = rendered_template
            await self._save_templates()

    async def _save_templates(self):
        async with aiofiles.open(self.templates_file, 'w', encoding='utf-8') as file:
            await file.write(yaml.dump({'templates': self.templates}))

    def _load_templates(self):
        logger.info(f'Templates cache location: {self.templates_file}')
        if not os.path.exists(self.templates_file):
            return {}

        with open(self.templates_file, encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data.get('templates', {})


class CachingTemplateRenderer:
    """Renders Wiki template from markup into actual text.

    Rendered templates are saved to a file for future usage
    """

    WIKI_API_URL = 'https://en.wikipedia.org/w/api.php'
    MARKER_START = 'WIKIDPOSTART__'
    MARKER_END = '__WIKIDPOEND'

    TEMPLATES_FILE = 'rendered_templates.yml'
    INVALID_TEMPLATES_FILE = 'invalid_templates.yml'

    def __init__(self):
        """Initialize templates renderer"""
        self.max_attempts = 5
        self.backoff_factor = 1  # initial backoff factor in seconds
        self.backoff_multiplier = 2  # double the backoff factor on each retry
        self.session = None  # aiohttp session
        self.semaphore = asyncio.Semaphore(3)

        self.template_cache = TemplatesCache(CachingTemplateRenderer.TEMPLATES_FILE)
        self.invalid_template_cache = TemplatesCache(CachingTemplateRenderer.INVALID_TEMPLATES_FILE)

        self.search_pattern = re.compile(r'WIKIDPOSTART__(.*?)__WIKIDPOEND')
        self.style_regex = re.compile(r'<style data-mw-deduplicate="TemplateStyles[^>]*>.*?</style>', flags=re.DOTALL)
        self.sup_regex = re.compile(r'<sup[^>]*>.*?</sup>', flags=re.DOTALL)
        self.html_tags_regex = re.compile(r'<[^>]+>')
        self.unwanted_class_regex = re.compile(
            # classes indicating image/map/etc.
            r'(<[^>]*\bclass=".*?(locmap|thumb|error|mw-message-box-error|mw-ext-score).*?"[^>]*>)|'
            # classes indicating non-existing template
            r'(<a[^>]*\bclass="new"[^>]*\btitle="Template[^"]*"[^>]*>)',
            flags=re.DOTALL,
        )

        self.requests_send = 0

    async def process_templates(self, raw_markup: str) -> tuple[bool, str | None]:
        """Process wikipedia markup and replace templates with real values

        Args:
            raw_markup: the raw template
        Returns:
            String representing the rendered template, or None if the template couldn't be rendered
        """
        wikicode = mwp.parse(raw_markup)
        text = []
        any_template_rendered = False
        for n in wikicode.nodes:
            if not isinstance(n, mwp.nodes.Template):
                text.append(str(n))
                continue
            template_str = str(n)

            # check if template was previously rendered
            rendered_template = self.template_cache.get_template(template_str)
            if rendered_template:
                text.append(rendered_template)
                any_template_rendered = True
                continue

            # check if the template is invalid (i.e. can't be rendered)
            invalid_template = self.invalid_template_cache.get_template(template_str)
            if invalid_template is not None:
                logger.error(f'Template {template_str} was previously invalid')
                return True, None

            # render template via Wiki API, since the template wasn't previously processed
            rendered_template = await self._render_template(template_str)
            if rendered_template is not None:
                # save the rendered template to cache
                await self.template_cache.add_template(template_str, rendered_template)

                text.append(rendered_template)
                any_template_rendered = True
                continue
            else:
                # save the invalid template to cache
                await self.invalid_template_cache.add_template(template_str, '')

                logger.error(f'Template {template_str} could not be rendered')
                return True, None

        if any_template_rendered:
            return True, ''.join(text)
        else:
            # no changes in text
            return False, raw_markup

    async def _render_template(self, template: str) -> str | None:
        """Render template using Wikipedia API"""
        async with self.semaphore:  # no more than 3 requests at a time
            self.requests_send += 1

            # wrap the template with markers, so it can be extracted from the full HTML returned by Wikipedia API
            req_text = f'{CachingTemplateRenderer.MARKER_START}{template}{CachingTemplateRenderer.MARKER_END}'
            if len(req_text) > 8081:
                # Maximum request length allowed by Wikipedia API
                logger.error(f'Request text too long: {req_text}')
                return None

            # https://en.wikipedia.org/w/api.php?action=help&modules=parse
            params = {
                'action': 'parse',
                'format': 'json',
                'contentmodel': 'wikitext',
                'text': req_text,
                'prop': 'text|templates',
            }
            data = await self._get_response_with_retry(params)
            if data is None:
                return None

            text = data['parse']['text']['*']

            text = self._extract_first_text(text)
            logger.debug(f'Rendered template {template} = {text}')
            return text

    async def _get_response_with_retry(self, params) -> dict[str, Any] | None:
        """Send request to Wikipedia API. Handle rate limiting"""
        attempt = 0
        backoff = self.backoff_factor
        while attempt < self.max_attempts:
            try:
                async with self.session.get(self.WIKI_API_URL, params=params) as response:
                    if response.status == 200:
                        return await response.json()  # Assuming JSON response
                    elif response.status == 429:
                        # request was rate limited; retry after backoff
                        logging.warning(
                            f'Attempt {attempt + 1}: Wikipedia API returned error: {response.status}; '
                            f'reason: {response.reason}'
                        )
                        attempt += 1
                        await asyncio.sleep(backoff)
                        backoff *= self.backoff_multiplier
                    else:
                        logging.error(f'Wikipedia API returned {response.status}, reason: {response.reason}')
                        return None
            except aiohttp.ClientError as e:
                logging.error(f'HTTP request failed: {e}')
                return None

        logging.error('Max retries reached, failed to render template.')
        return None

    def _extract_first_text(self, text):
        """Extract rendered template from HTML"""
        match = self.search_pattern.search(text)
        if match:
            rendered_template = match.group(1)
            return self._decode_html(rendered_template)

        logger.warning(f'Failed to match {text}')
        return None

    def _decode_html(self, html_string: str) -> str | None:
        # Check for unwanted classes in tags (errors, images, maps, etc.)
        if self.unwanted_class_regex.search(html_string):
            return None

        # Remove <style> tags with specific attributes
        no_style_text = self.style_regex.sub('', html_string)
        # Remove content inside <sup> tags, including the tags themselves
        no_sup_text = self.sup_regex.sub('', no_style_text)
        # Now remove all remaining HTML tags
        plain_text = self.html_tags_regex.sub('', no_sup_text)
        # Unescape HTML characters
        text = html.unescape(plain_text)
        # replace non-breakable space
        return text.replace('\xa0', ' ')

    async def start_session(self):
        """Start a aiohttp session"""
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Close the aiohttp session"""
        await self.session.close()

    async def __aenter__(self):
        """Open a aiohttp session"""
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Close the aiohttp session"""
        await self.close_session()
