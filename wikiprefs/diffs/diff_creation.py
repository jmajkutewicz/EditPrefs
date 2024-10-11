import difflib
import multiprocessing
import re
from dataclasses import dataclass
from enum import Enum, auto

from wikiprefs.diffs.model import SectionDiff, SegmentDiff, TextDiff
from wikiprefs.diffs.tokenization import Tokenizer
from wikiprefs.markup.markup_processing import sections_to_dict

logger = multiprocessing.get_logger()


class DiffState(Enum):
    """Diff aggregation state"""

    ADDED = 1
    REMOVED = 2
    UNCHANGED = 3


class FragmentType(Enum):
    """Text fragment (e.g. line, substring) type"""

    TEXT = auto()
    TEMPLATE = auto()


@dataclass
class SectionText:
    """Article section text"""

    text: str
    sentences: list[str]


class SectionsComparator:
    """Comparator that finds differences between 2 versions of the same articles"""

    def __init__(self):
        """Initializes the comparator instance"""
        self.tokenizer = Tokenizer()
        self.newline_pattern = re.compile(r'(\n+)')

        self.single_sentence_templates = {'quote', 'bq', '"', 'bquote', 'blockquote', 'blockindent', 'bi', 'math'}
        self.template_name_pattern = re.compile(r'^\{\{\s*([^|}]+)')

    def tokenize_sections(self, sections: [tuple[str, str]]) -> [tuple[str, SectionText]]:
        """Splits the section text into separate sentences

        Args:
            sections: list of tuples (section title, section text)

        Returns:
            List of tuples (section title, SectionText). SectionText contains both raw and tokenized text

        Raises:
            ProblematicTokenizationPatternException: if the text contains pattern that prevents tokenization
            TooLongSectionException: if a section text is too long to tokenize
        """
        sections_tokenized = []
        for section_title, text in sections:
            text_tokenized = self._tokenize_text(text)
            sections_tokenized.append((section_title, SectionText(text, text_tokenized)))

        return sections_tokenized

    def compare_sections(
        self,
        sections_v1_old: [tuple[str, SectionText]],
        sections_v2_fixed: [tuple[str, SectionText]],
        context_lines: int = 3,
    ) -> [SectionDiff]:
        """Compare sections of two versions of the article and find differences.

        The comparison is done on a section level. The text of 2 versions of the same section is compared
        using :func:`difflib.unified_diff`

        Comparison details:
        - old sections removed from v2 are ignored
        - new sections added in v2 are ignored
        - identical sections are ignored
        - changes for modified parts of the text are grouped together if they are adjacent

        Args:
            sections_v1_old: tokenized text (as returned by :func:`.tokenize_sections`) of the old article version
            sections_v2_fixed: tokenized text (as returned by :func:`.tokenize_sections`) of the new article version
            context_lines: number of context lines for unified diffs (Default value = 3)

        Returns:
            List of :class:`SectionDiff`
        """
        v2_sections_dict = sections_to_dict(sections_v2_fixed)
        sections_diffs = []

        # iterate over all sections from v1. This will ignore new sections added in v2
        for section_title, v1_section_text in sections_v1_old:
            if section_title not in v2_sections_dict:
                # skip sections removed in v2
                continue

            v2_section_text = v2_sections_dict[section_title]
            if v1_section_text.text == v2_section_text.text:
                continue

            v1_sentences = v1_section_text.sentences
            v2_sentences = v2_section_text.sentences

            # Create diffs using unified diffs.
            # Unified diffs are a compact way of showing line changes and a few  of context
            diff_lines = difflib.unified_diff(v1_sentences, v2_sentences, n=context_lines)

            all_segments_diffs = []
            segment_diffs = None
            curr_diff = TextDiff()
            diff_state = DiffState.UNCHANGED

            for line in diff_lines:
                if line.startswith('---') or line.startswith('+++'):
                    # ignore diff control lines (see difflib.unified_diff for details)
                    continue
                if len(line) == 0:
                    # ignore empty lines
                    continue
                if line.startswith('@@'):
                    # start of a new diff segment
                    if segment_diffs:
                        all_segments_diffs.append(SegmentDiff(segment_diffs))
                    segment_diffs = []
                    continue

                if (diff_state == DiffState.ADDED and not line.startswith('+')) or (
                    diff_state == DiffState.REMOVED and not (line.startswith('+') or line.startswith('-'))
                ):
                    # single diff ends if:
                    #  previous line was added, and the current line is unchanged or removed
                    #  previous line was removed and current line is unchanged
                    segment_diffs.append(curr_diff)
                    curr_diff = TextDiff()

                if line.startswith('-'):
                    diff_state = diff_state.REMOVED

                    # line present in v1 that was either changed or completely removed in v2
                    if curr_diff.v1 is None:
                        curr_diff.v1 = line[1:]
                    else:  # aggregate all subsequent changed/deleted lines
                        diff_text = line[1:]
                        if not curr_diff.v1[-1].isspace() and not diff_text.isspace():
                            # spacy tokenization removes trailing spaces so this is best effort attempt to restore them
                            curr_diff.v1 += ' '
                        curr_diff.v1 += diff_text
                elif line.startswith('+'):
                    diff_state = diff_state.ADDED

                    # line present in v2 that was either changed compared to v1 or is a completely new line
                    if curr_diff.v2 is None:
                        curr_diff.v2 = line[1:]
                    else:  # aggregate all subsequent changed/added lines
                        diff_text = line[1:]
                        if not diff_text.isspace():
                            diff_text = diff_text.lstrip()
                        if not curr_diff.v2[-1].isspace() and not diff_text.isspace():
                            # spacy tokenization removes trailing spaces so this is best effort attempt to restore them
                            curr_diff.v2 += ' '
                        curr_diff.v2 += diff_text
                else:
                    diff_state = diff_state.UNCHANGED
                    # unchanged line (just append it for context)
                    segment_diffs.append(line[1:])

            if diff_state == DiffState.ADDED or diff_state == DiffState.REMOVED:
                segment_diffs.append(curr_diff)
            if segment_diffs:
                all_segments_diffs.append(SegmentDiff(segment_diffs))
            sections_diffs.append(SectionDiff(section_title, all_segments_diffs))
        return sections_diffs

    def _tokenize_text(self, text: str) -> [str]:
        sentences = []
        lines = [s for s in self.newline_pattern.split(text) if s]
        for line in lines:
            line_tokenized = self._tokenize_line(line)
            if line.startswith('*'):
                # keep list item as a single sentence
                line_tokenized = ''.join(line_tokenized)
                sentences.append(line_tokenized)
            else:
                sentences.extend(line_tokenized)

        sentences = self._fix_tokenized_math(sentences)
        return sentences

    def _tokenize_line(self, text: str) -> [str]:
        sentences = []
        fragments = self._split_on_templates(text)
        for f_type, f in fragments:
            if f_type == FragmentType.TEMPLATE:
                # template (e.g. quote) that we want to treat as a single line during diff creation
                sentences.append(f)
            else:
                doc = self.tokenizer.tokenize(f)
                f_sentences = [sent.text for sent in doc.sents]
                f_sentences = self._fix_tokenized_math(f_sentences)
                sentences.extend(f_sentences)
        return sentences

    def _split_on_templates(self, text: str) -> list[tuple[FragmentType, str]]:
        """Splits a line into text and templates that should be treated as a single sentence

        The templates that are treated as a single sentence are quotes({{quote|...}}) and math ({{math|}})
        This allows to treat a quote as a single "line" during diff creation and avoid breaking it up
        """
        result = []
        stack = []
        i = 0
        length = len(text)

        while i < length:
            if i + 2 <= length and text[i : i + 2] == '{{':
                # Push the position and the current index into the stack
                stack.append((i, len(result)))
                i += 2
                continue

            if i + 2 <= length and text[i : i + 2] == '}}':
                if stack:
                    start, result_index = stack.pop()
                    # When the stack is empty, it means we have an outermost template
                    if not stack:
                        template_text = text[start : i + 2]
                        template_type = self.template_name_pattern.search(template_text)
                        if template_type:
                            template_type = template_type.group(1).strip().lower().replace(' ', '')
                            if template_type in self.single_sentence_templates:
                                # Append text before the template if it is the initial text
                                # or different from the last segment added
                                if start > 0 and (not result or result[-1][1] != text[:start]):
                                    result.append((FragmentType.TEXT, text[:start]))
                                # Insert the quote
                                result.append((FragmentType.TEMPLATE, template_text))

                                # Move past any additional spaces if they follow the template
                                while i + 2 < length and text[i + 2] == ' ':
                                    i += 1
                                text = text[i + 2 :]
                                i = -1  # Reset index to start of the new segment
                                length = len(text)
                                continue
                i += 2
                continue

            i += 1

        # Append the remaining text as is, preserving whitespace
        if text:
            result.append((FragmentType.TEXT, text))

        return result

    def _fix_tokenized_math(self, tokens):
        """Re-joins <math> and </math> if they were split during tokenization by spacy

        This allows to treat a quote as a single "line" during diff creation and avoid breaking it up
        """
        fixed_tokens = []
        in_math = False
        math_content = []

        for token in tokens:
            if '<math' in token:
                # Start collecting math content
                if in_math:
                    # If already in math mode, just continue collecting
                    math_content.append(token)
                else:
                    # Start a new math content collection
                    in_math = True
                    math_content = [token]
            elif '</math>' in token:
                # Append current token to math content and close math block
                math_content.append(token)
                fixed_tokens.append(' '.join(math_content))
                math_content = []
                in_math = False
            elif in_math:
                # Continue collecting math content if within a math block
                math_content.append(token)
            else:
                # Normal token, not within a math block
                fixed_tokens.append(token)

        # Handle case where there's no closing </math> tag
        if in_math:
            fixed_tokens.extend(math_content)

        return fixed_tokens
