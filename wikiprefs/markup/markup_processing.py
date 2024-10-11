import logging
import multiprocessing
import os.path
import re
from enum import Enum, auto
from pathlib import Path
from typing import Any

import mwparserfromhell as mwp
import mwparserfromhell.nodes as mwp_nodes
import yaml

logger = multiprocessing.get_logger()


class WikiDuplicatedSectionException(Exception):
    """Two (or possibly more) sections with same name (up to whole path in page hierarchy)

    According to https://en.wikipedia.org/wiki/Help:Section#Creation_and_numbering_of_sections
    names of sections (including subsections) should be unique on a page
    """

    def __init__(self, message: str, section_title: str):
        """Initializes the exception instance"""
        super().__init__(message)
        self.section_title = section_title


class WikiBrokenMarkupException(Exception):
    """The Wiki markup is broken to the point where it can't be parsed correctly by mwparserfromhell"""

    def __init__(self, message: str):
        """Initializes the exception instance"""
        super().__init__(message)


class HeadingHandler:
    """Section heading handler. Keeps track of the whole section hierarchy"""

    # Artificial name for introduction section (avoids conflicts with other sections names)
    INTRO_SECTION = 'default__wiki__article__introduction__heading'

    def __init__(self):
        """Initializes the heading handler instance"""
        self.current_level = 0
        self.section_path = []

    def next_section(self, heading: mwp_nodes.Heading | None) -> tuple[str, str]:
        """Processing enters new section

        Args:
            heading: section heading node
        Returns:
            Tuple: section title; section full path

            The section full path, is the hierarchy of all parent sections joined with //,
            e.g.: 'History // 1980 // Details'
        Raises
            WikiDuplicatedSectionException: if the section hierarchy is completely broken
        """
        if heading is None:
            # first section has no explicit heading, so we set artificial on with level 2 (lowest allowed level)
            title = HeadingHandler.INTRO_SECTION
            self.current_level = 2
            self.section_path.append(title)
        else:
            title = heading.title.strip()
            level = heading.level
            if level < 2:
                # according to Wiki docs, the lowest allowed level is 2
                logger.debug(f'Heading "{title}" has too low level: {level}.')
                level = 2

            if level < self.current_level:
                # parsing went from higher level (more nested section) to lower level (parent section sibling)
                self.current_level = level
                self.section_path = self.section_path[: level - 1]
                self.section_path[-1] = title
            elif level == self.current_level:
                # sibling section (same level in hierarchy)
                self.section_path[-1] = title
            else:  # level > self.current_level
                # parsing going into lower section (higher level)
                if self.current_level + 1 != level:
                    # more than 1 level jump - fill with empty sections
                    for _ in range(0, level - self.current_level - 1):
                        self.section_path.append('')
                self.section_path.append(title)
                self.current_level = level

        return title, ' // '.join(self.section_path)


class SelfContainedTemplatesNodeType(Enum):
    """Node type for self-contained templates removal (i.e. templates that constitute a whole paragraph)"""

    TEMPLATE = auto()
    BREAK = auto()
    BREAK_THEN_TEXT = auto()
    TEXT_THEN_BREAK = auto()
    OTHER = auto()

    @staticmethod
    def get_node_type(node: mwp_nodes.Node):
        """Returns the node type"""
        if isinstance(node, mwp_nodes.Template):
            return SelfContainedTemplatesNodeType.TEMPLATE
        if isinstance(node, mwp_nodes.Text):
            text = node.value
            if _is_paragraph_break(text):
                return SelfContainedTemplatesNodeType.BREAK
            if text.startswith('\n\n'):
                return SelfContainedTemplatesNodeType.BREAK_THEN_TEXT
            if text.endswith('\n\n'):
                return SelfContainedTemplatesNodeType.TEXT_THEN_BREAK
        if isinstance(node, mwp_nodes.Heading):
            return SelfContainedTemplatesNodeType.BREAK
        return SelfContainedTemplatesNodeType.OTHER


class SelfContainedTemplatesRemovalState(Enum):
    """Self-contained templates removal state"""

    NONE = auto()
    IN_TEXT = auto()


class WikiMarkupParser:
    """Wikipedia Markup parser optimized for comparing different revision of the same article on a section level"""

    IGNORED_TEMPLATES_FILE = 'ignored_templates.yml'
    IGNORED_SECTIONS_FILE = 'ignored_sections.yml'

    def __init__(self):
        """Initializes the parser instance"""
        self.template_pattern = re.compile(r'\{\{([\s\S]*?)}}')
        self.wiki_section_pattern = re.compile(r'^=.*=(?:<!--.*?-->)?\s*$', re.MULTILINE)
        self.ref_pattern = re.compile(r'<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)
        self.comment_pattern = re.compile(r'^<!--.*?-->\s*[\r\n]?', re.DOTALL | re.MULTILINE)

        self.node_strip_kwargs = {
            'normalize': True,
            'collapse': True,
            'keep_template_params': False,
        }
        self.ignored_templates = set()
        self.mapped_templates = dict()
        self.multiline_templates = dict()
        self._load_ignored_templates()

        self.references_sections = set()
        self.notes_sections = set()
        self._load_ignored_sections()

    def _load_ignored_templates(self) -> None:
        ignored_templates_file = os.path.join(
            Path(__file__).parent.parent, 'resources', WikiMarkupParser.IGNORED_TEMPLATES_FILE
        )

        with open(ignored_templates_file, encoding='utf-8') as file:
            data = yaml.safe_load(file)

        ignored_templates = data.get('ignored', {})
        for v in ignored_templates.values():
            self.ignored_templates.update(map(lambda s: s.lower().strip(), v))
        logger.debug(f'Loaded {len(self.ignored_templates)} ignored templates')

        self.mapped_templates = data.get('mapped', {})
        logger.debug(f'Loaded {len(self.mapped_templates)} mapped templates')

        multiline_templates = data.get('multiline', [])
        for t in multiline_templates:
            t_end = t.get('end').lower()
            t_starts = t.get('start', [])
            for t_start in t_starts:
                self.multiline_templates[t_start.lower()] = t_end
        logger.debug(f'Loaded {len(self.multiline_templates)} multiline templates')

    def _load_ignored_sections(self):
        ignored_sections_file = os.path.join(
            Path(__file__).parent.parent, 'resources', WikiMarkupParser.IGNORED_SECTIONS_FILE
        )
        with open(ignored_sections_file, encoding='utf-8') as file:
            data = yaml.safe_load(file)

        references_sections = data.get('references')
        self.references_sections = set(map(lambda s: s.lower(), references_sections))
        logger.debug(f'Loaded {len(self.references_sections)} ignored resources sections')

        notes_sections = data.get('footnotes')
        self.notes_sections = set(map(lambda s: s.lower(), notes_sections))
        logger.debug(f'Loaded {len(self.notes_sections)} ignored notes sections')

    def parse_wiki_markup(self, raw_markup: str, ignore_errors: bool = False) -> [tuple[str, str]]:
        """Splits a raw Wikipedia Article markup into separate sections and removes markup.

        A parent section contains only its text, and doesn't contain its subsections.
        Common meta-sections (like references, bibliography, etc.) are also removed.

        Returns:
            list of tuples (section title, section cleaned text)

        Raises:
            WikiBrokenMarkupException: if the markup couldn't be parsed
            WikiDuplicatedSectionException: if the section hierarchy is completely broken
        """
        wikicode = self._preparse_markup(raw_markup, ignore_errors)

        # collect sections with cleaned text
        sections = []
        sections_paths = set()
        headings_handler = HeadingHandler()
        for section in wikicode.get_sections(include_lead=True, flat=True):
            headings = section.filter_headings()
            title, full_title_path = headings_handler.next_section(headings[0] if headings else None)

            title_normalized = title.strip().lower()
            if title_normalized in self.notes_sections:
                # ignore notes
                continue
            if title_normalized in self.references_sections:
                # stop processing after first references section (most of the time there's no content after it)
                break

            if full_title_path in sections_paths:
                # according to https://en.wikipedia.org/wiki/Help:Section#Creation_and_numbering_of_sections
                # names of sections (including subsections) should be unique on a page
                raise WikiDuplicatedSectionException(f'Duplicate section: {full_title_path}', full_title_path)
            sections_paths.add(full_title_path)

            text = []
            for n in section.nodes:
                if isinstance(n, mwp_nodes.Heading):
                    continue

                if isinstance(n, mwp_nodes.Tag):
                    if n.tag == 'ref' or n.tag == 'table':
                        # ignore <ref> and tables
                        continue
                    elif n.tag == 'i':
                        # citations (i.e. "text", not sources citations) are represented as <i> tag by mwparserfromhell
                        node_text = self._strip_i_tag(n)
                        if node_text:
                            text.append(f'"{node_text}"')
                        continue
                    elif n.tag == 'math':
                        # <math> is not rendered by default by mwparserfromhell as '<math>' is in mwp INVISIBLE_TAGS
                        # it contains latex math notation, so we keep it as it is
                        n._attrs = []  # remove any html attributes from <math> tag (e.g. <math display=block>)
                        node_text = str(n)
                        if node_text:
                            text.append(str(node_text))
                        continue

                    if n.tag == 'li':
                        # list; add * to the begging of the line
                        text.append('*')
                    elif n.tag == 'dd':
                        # indentation (represented as : in markup)
                        text.append(' ')

                if isinstance(n, mwp_nodes.Template):
                    template_name = n.name.lower().strip()
                    if self._is_ignored_template(template_name):
                        logger.debug(f'Ignoring template {template_name}')
                        continue

                    if template_name in self.mapped_templates and not n.params:
                        template_text = self.mapped_templates[template_name]
                        logger.debug(f'Using {template_text} for template {template_name}')
                    else:
                        # keep the template, it will be later rendered to proper text
                        template_text = str(n)
                    text.append(template_text)
                    continue

                node_text = n.__strip__(**self.node_strip_kwargs)
                if node_text:
                    text.append(str(node_text))

            # try to remove excess whitespace
            text = ''.join(text).strip('\n')
            while '\n\n\n' in text:
                text = text.replace('\n\n\n', '\n\n')
            # replace non-breakable space
            text = text.replace('\xa0', ' ').strip()
            # remove <ref> tags
            text = re.sub(self.ref_pattern, '', text).strip()

            sections.append((full_title_path, text))

        return sections

    def _is_ignored_template(self, template_name: str) -> bool:
        return any(template_name.startswith(ignored_template) for ignored_template in self.ignored_templates)

    def _validate_wikicode(self, raw_markup: str, wikicode: mwp.wikicode) -> None:
        """Sanity check for validating if mwparserfromhell processed the markup correctly.

        If the number of section returned by mwparserfromhell is different from sections detected by regex,
        then processing fatally failed, and we can't process the article further

        Raises:
            WikiBrokenMarkupException: if the markup couldn't be parsed by mwparserfromhell
        """
        code_headings = wikicode.get_sections(include_lead=False)
        code_headings_count = len(code_headings)

        regex_headings = self.wiki_section_pattern.findall(raw_markup)
        regex_headings_count = len(regex_headings)

        if code_headings_count != regex_headings_count:
            msg = f'Mismatching headings count: {len(code_headings)}(mwp) vs. {len(regex_headings)}(regex)'
            logger.debug(msg)
            raise WikiBrokenMarkupException(msg)

    def _preparse_markup(self, raw_markup: str, ignore_errors) -> mwp.wikicode:
        """Clean up markup before it's split into sections

        Remove self-contained templates, file and images links, cleans up headings, etc.
        """
        # remove comments using regex to avoid leaving emtpy lines
        raw_markup = self.comment_pattern.sub('', raw_markup)

        wikicode = mwp.parse(raw_markup)
        if not ignore_errors:
            self._validate_wikicode(raw_markup, wikicode)

        # remove comments
        for comment in wikicode.filter_comments():
            wikicode.remove(comment)

        # remove file, category and image wikilinks
        for wl in wikicode.filter_wikilinks():
            wl_title = str(wl.title)
            wl_title = wl_title.lower().strip()

            if wl_title.startswith('file:') or wl_title.startswith('category:') or wl_title.startswith('image:'):
                try:
                    wikicode.remove(wl)
                except ValueError:
                    pass  # happens for some weird reason

        nodes = wikicode.nodes

        # remove all templates before first text
        text_start = 0
        for i, n in enumerate(nodes):
            if isinstance(n, mwp_nodes.Text) and not _is_paragraph_break(n.value):
                break
            if isinstance(
                n,
                mwp_nodes.Heading | mwp_nodes.Tag | mwp_nodes.HTMLEntity | mwp_nodes.Wikilink | mwp_nodes.ExternalLink,
            ):
                break
            if isinstance(n, mwp_nodes.Template) and i + 1 < len(nodes):
                next_note = nodes[i + 1]
                if not (isinstance(next_note, mwp_nodes.Text) and next_note.value.startswith('\n')):
                    # template node next sibling is not a paragraph break -> it's the first node of the article's text
                    break

            text_start += 1
        nodes = nodes[text_start:]

        # remove multi-line template (e.g. {{Nat fs start}} that is just a table underneath)
        i = 0
        while i < len(nodes):
            n = nodes[i]

            if not isinstance(n, mwp_nodes.Template):
                i += 1
                continue

            template_name = n.name.lower().strip()
            if template_name not in self.multiline_templates:
                i += 1
                continue

            template_start = template_name
            template_end = self.multiline_templates[template_name]
            while True and i < len(nodes):
                del nodes[i]

                n = nodes[i]
                if isinstance(n, mwp_nodes.Heading):
                    # next section, multi-line template wasn't closed
                    logger.debug(f'Unclosed multi line template {template_start}')
                    break

                if not isinstance(n, mwp_nodes.Template):
                    continue

                template_name = n.name.lower().strip()
                if template_name == template_end:
                    del nodes[i]
                    break

        # remove lines that contains only templates (i.e. self-contained templates like info boxes)
        # a template is removed iff:
        # * it's preceded by paragraph break/sections start/another templates
        #   that were preceded by paragraph break/sections start,
        # * and it's proceeded by paragraph break/sections end/another templates
        #   that will be proceeded by paragraph break/sections end
        nodes_count = len(nodes)
        state = SelfContainedTemplatesRemovalState.NONE
        current_templates = []  # stores parsed templates, until it's decided if they should be removed or kept
        nodes_to_remove = []  # id of nodes to remove (allows to avoid modifying list while iterating)
        for i in range(nodes_count + 1):
            if i < nodes_count:
                node_type = SelfContainedTemplatesNodeType.get_node_type(nodes[i])
            else:
                # +1 iteration to handle templates at the end of the wikicode
                node_type = SelfContainedTemplatesNodeType.BREAK

            match state:
                case SelfContainedTemplatesRemovalState.NONE:
                    match node_type:
                        case SelfContainedTemplatesNodeType.TEMPLATE:
                            # section or paragraph start
                            current_templates.append(i)
                        case SelfContainedTemplatesNodeType.BREAK:
                            # paragraph end
                            if current_templates:
                                nodes_to_remove.extend(current_templates)
                                current_templates = []
                        case SelfContainedTemplatesNodeType.BREAK_THEN_TEXT:
                            # paragraph end, then new paragraph with text start
                            # (mw parser from hell concatenates \n\n (paragraph break) and text right after it)
                            if current_templates:
                                nodes_to_remove.extend(current_templates)
                                current_templates = []
                            state = SelfContainedTemplatesRemovalState.IN_TEXT
                        case SelfContainedTemplatesNodeType.TEXT_THEN_BREAK:
                            # paragraph started and immediately finished. Previous template were probably  part of it
                            current_templates = []
                        case SelfContainedTemplatesNodeType.OTHER:
                            # some text
                            current_templates = []
                            state = SelfContainedTemplatesRemovalState.IN_TEXT
                case SelfContainedTemplatesRemovalState.IN_TEXT:
                    match node_type:
                        case SelfContainedTemplatesNodeType.BREAK:
                            # paragraph end
                            state = SelfContainedTemplatesRemovalState.NONE
                        case SelfContainedTemplatesNodeType.TEXT_THEN_BREAK:
                            state = SelfContainedTemplatesRemovalState.NONE
                        case SelfContainedTemplatesNodeType.TEMPLATE:
                            # template inside text
                            pass
                        case SelfContainedTemplatesNodeType.OTHER:
                            # text continuation in a text
                            pass

        if logger.isEnabledFor(logging.DEBUG):
            template_to_remove = [nodes[i] for i in nodes_to_remove]
            template_to_remove = [t.name if isinstance(t, mwp_nodes.Template) else str(t) for t in template_to_remove]
            logger.debug(f'Removing self-contained templates: {template_to_remove}')

        nodes = [node for i, node in enumerate(nodes) if i not in nodes_to_remove]

        wikicode.nodes = nodes

        # remove templates from headings:
        headings = wikicode.filter_headings()
        for h in headings:
            nodes_to_remove = []
            title_nodes = h.title.nodes
            for i, node in enumerate(title_nodes):
                if isinstance(node, mwp_nodes.Template):
                    nodes_to_remove.append(i)
            if nodes_to_remove:
                for i in reversed(nodes_to_remove):
                    del title_nodes[i]

        return wikicode

    def _strip_i_tag(self, n: mwp_nodes.Tag) -> str:
        """If there is a tag inside the quotes (i tag) we should keep it (it could be text wrapped in a template

        For now, only the simplest case is handled
        """
        # FIXME handle other cases:
        contents = n.contents.nodes
        if len(contents) == 1:
            c = contents[0]
            if isinstance(c, mwp_nodes.Template):
                return str(c)

        return n.__strip__(**self.node_strip_kwargs)


def _is_paragraph_break(s):
    return s.isspace() and '\n' in s


def sections_to_dict(sections: [tuple[str, Any]]) -> dict[str, Any]:
    """Convert list (section_name, section text) to dictionary"""
    sections_dict = {s[0]: s[1] for s in sections}
    return sections_dict
