import multiprocessing
import re

import spacy
from spacy.tokens import Doc

logger = multiprocessing.get_logger()


class ProblematicTokenizationPatternException(Exception):
    """Exception that occurs when a pattern that might cause Spacy to be stuck is detected

    For example text ending with 1000s of ! marks causes Spacy to hang
    """

    def __init__(self, message: str, text: str):
        """Initializes the exception instance

        Args:
            message: exception message
            text: section text that caused the exception
        """
        super().__init__(message)
        self.text = text


class TooLongSectionException(Exception):
    """Exception that occurs when a section's text is too long to tokenize

    Spacy tokenizer require roughly 1GB of temporary memory per 100,000 characters in the input.
    So long texts may cause memory allocation errors, and Spacy sets the max allowed text length to 1_000_000 by default
    """

    def __init__(self, message: str, section_title: str):
        """Initializes the exception instance

        Args:
            message: exception message
            section_title: section title that caused the exception
        """
        super().__init__(message)
        self.section_title = section_title


class Tokenizer:
    """Text tokenizer (wrapper around spacy)"""

    def __init__(self):
        """Initializes the tokenizer instance"""
        self.tokenizer = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'lemmatizer'])
        logger.debug(f'Max tokenization length: {self.tokenizer.max_length}')

        # template for detecting repeated characters, that might cause spacy to get stuck
        self.problematic_patterns = {
            'excessive_punctuation': re.compile(r'[?.!]{1000,}'),
            'repeated_characters': re.compile(r'(.)\1{1000,}'),
            'repeated_non_alpha': re.compile(r'([^\w\s]{1,3})\1{100,}'),
        }

    def validate_text(self, section_title: str, text: str) -> None:
        """Validate if text can be tokenized

        Args:
            section_title: section title that caused the exception
            text: text to tokenize

        Raises:
            TooLongSectionException: if a section text is too long to tokenize
        """
        if len(text) > self.tokenizer.max_length:
            raise TooLongSectionException(
                f'Section {section_title} is too long to tokenize ({len(text)})', section_title
            )

    def tokenize(self, text: str) -> Doc:
        """Tokenizes the given text using spacy

        Args:
            text: text to tokenize

        Returns:
            spacy Doc object

        Raises:
            ProblematicTokenizationPatternException: if the text contains pattern that prevents tokenization
            TooLongSectionException: if a section text is too long to tokenize
        """
        if self._has_problematic_patterns(text):
            # text can't be tokenized by Spacy (it will hang indefinitely)
            raise ProblematicTokenizationPatternException('Problematic pattern detected in text', text)

        return self.tokenizer(text)

    def _has_problematic_patterns(self, text):
        return any(regex.search(text) for regex in self.problematic_patterns.values())
