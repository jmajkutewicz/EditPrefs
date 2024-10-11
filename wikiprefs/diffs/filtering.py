import enum
import multiprocessing
import os
import re
from collections.abc import Callable

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer

from wikiprefs.diffs.tokenization import ProblematicTokenizationPatternException, Tokenizer, TooLongSectionException

logger = multiprocessing.get_logger()


class CommentFilter:
    """Filter that check if revision is a neutral point of view (NPOV) improvement"""

    # wiki tags that mark text neutralization
    NPOV_TAGS = {
        'WP:NPOV',
        'WP:UNDUE',
        'WP:undue weight',
        'WP:CONSENSUS',
        'WP:CON',
        'WP:FRINGE',  # Fringe theories
        'WP:FRNG',
        'WP:DISPUTED',  # Accuracy dispute
        'WP:DUBIOUS',
        'WP:AD',
        'WP:NOTOPINION',  # not a soapbox or means of promotion
        'WP:NOTSCANDAL',
        'WP:SOAP',
        'WP:SOAPBOX',
        'WP:CRYSTAL',
        'WP:CRYION',
        'WP:FUTURESTALBALL',
        'WP:NOTCRYSTAL',
        'WP:RUMOUR',
        'WP:RUMOR',
        'WP:SPECULAT',
    }

    FEATURED_ARTICLES_TAGS = {
        'WP:FA',
        'WP:FACR',
        'WP:FACRITERIA',
        'WP:WIAFA',
        'WP:WBA',  # Writing better articles
        'WP:MTAU',  # Make technical articles understandable
        'WP:TECHNICAL',
        'WP:OVERSIMPLIFY',
        "WP: Don't lie",
        'WP:DNTL',
        'WP:LIE',
        # https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style
        'MOS:PUFFERY',  # promote the subject of an article
        'WP:PUFFERY',
        'MOS:PEACOCK',
        'WP:PEACOCK',
        'MOS:FLOWERY',
        'WP:FLOWERY',
        'MOS:WEASEL',  # Unsupported attributions
        'WP:WEASEL',
        'MOS:AWW',
        'WP:AWW',
        'MOS:ACCUSED',  # Expressions of doubt
        'WP:ACCUSED',
        'MOS:ALLEGED',
        'WP:ALLEGED',
        'MOS:DOUBT',
        'WP:DOUBT',
        'MOS:SCAREQUOTES',
        'WP:SCAREQUOTES',
        'MOS:CONFUSE',  # Easily confused terms
        'WP:CONFUSE',
        'MOS:ARAB',
        'WP:ARAB',
    }

    VANDALISMS = {
        'vandalism',
        'WP:VAND',
        'WP:VD',
        'WP:VANDAL',
    }

    class _CheckType(enum.Enum):
        NPOV = enum.auto()
        FA = enum.auto()

    def __init__(self):
        """Initialize filter instance"""
        self.revert_indicators = ['reverted', 'revert', 'undo', 'undid', 'rollback']
        self.npov_re = r'(?:^|\s|:)n?pov(?:$|\s|\]|/)'
        self.npov_keywords = ['npov', 'neutralized bias', 'removed pov', 'impartial', 'fairness']

        self.npov_tags_re = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in CommentFilter.NPOV_TAGS) + r')\b(?![a-zA-Z])', re.IGNORECASE
        )
        self.fa_tags_re = re.compile(
            r'\b(' + '|'.join(re.escape(w) for w in CommentFilter.FEATURED_ARTICLES_TAGS) + r')\b(?![a-zA-Z])',
            re.IGNORECASE,
        )

    def is_fa_edit(self, comment: str | None) -> bool:
        """Check if the comment suggest feature article text improvement

        Args:
            comment: edit comment for the revision
        """
        return self._check_comment(comment, CommentFilter._CheckType.FA)

    def is_npov_edit(self, comment: str | None) -> bool:
        """Check if the comment suggest NPOV edit (e.g. neutralizing text)

        Args:
            comment: edit comment for the revision
        """
        return self._check_comment(comment, CommentFilter._CheckType.NPOV)

    def _check_comment(self, comment: str | None, check_type: _CheckType) -> bool:
        if not comment:
            return False

        comment = comment.lower()

        # ignore reverts (they are often low quality, so we skip them)
        if any(indicator in comment for indicator in self.revert_indicators):
            return False
        if comment.startswith('rv '):
            return False

        # check vandalism (again, usually low quality edits)
        if any(v.lower() in comment for v in CommentFilter.VANDALISMS):
            return False

        if check_type == CommentFilter._CheckType.FA:
            return self.fa_tags_re.search(comment) is not None
        elif check_type == CommentFilter._CheckType.NPOV:
            # check POV
            if self._is_pov(comment):
                return True
            return self.npov_tags_re.search(comment) is not None
        else:
            return False

    def _is_pov(self, comment: str) -> bool:
        if re.search(self.npov_re, comment) is not None:
            return True
        return any(keyword in comment for keyword in self.npov_keywords)


class LengthFilter:
    """Filter that checks if the text will fit into a LLM model (in terms of prompt length).

    By default, uses LLama2 config
    """

    def __init__(self, max_length: int = 1024, tokenizer_name: str = 'meta-llama/Llama-2-7b-chat-hf'):
        """Initialize filter instance

        Args:
            max_length: maximum length of the tokenized text
            tokenizer_name: huggingface tokenizer name
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def is_too_long(self, old_text: str, new_text: str) -> bool:
        """Check if the text is too long to fit into a LLM model"""
        longer_text = new_text if len(new_text) > len(old_text) else old_text
        tokenized_text = self.tokenizer.tokenize(longer_text)
        length = len(tokenized_text)

        return length > self.max_length


class OutliersFilter:
    """Filter that removes outliers, based on general dataset statistics.

    Rows are removed if:
    - any text is empty, or both new and old text are the same
    - the relative difference (in %) between new and old text is too long (top 99th percentile)
    - the edit distance between old and new text is too small (less than 4 characters)
    - the edit distance w.r.t. old text is lower than 10%
    """

    def __init__(
        self,
        comment_filter: Callable[[str | None], bool],
        max_relative_length_difference_quantile: float,
        filter_out_simple_edits: bool,
    ):
        """Initialize the OutliersFilter"""
        self.whitespace_re = re.compile(r'\s+')
        self.min_edit_distance = 4
        self.min_words_count = 5
        self.minimal_max_relative_length_difference = 150.0
        self.maximal_max_relative_length_difference = 200.0
        self.max_relative_length_difference_quantile = max_relative_length_difference_quantile

        self.comment_filter = comment_filter
        self.filter_out_simple_edits = filter_out_simple_edits

    def filter_diffs(self, df) -> pd.DataFrame:
        """Removes rows that are outliers"""
        df = self._filter_empty_text(df)
        df = self._filter_same_text(df)
        df = self._filter_short_text(df)
        df = self._filter_by_relative_length_difference(df)

        # Calculate levenshtein distance
        df['edit_distance'] = df.apply(self._calculate_edit_distance, axis=1)
        logger.info(f'edit distance quantiles: \n{df["edit_distance"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])}')

        df = self._filter_by_edit_distance(df)
        df = self._filter_by_relative_edit_distance(df)

        return df

    def _filter_by_relative_edit_distance(self, df):
        """Remove rows where the edit distance w.r.t. old text is lower than 10%, and it's not an NPOV edit"""
        df_len = df.shape[0]
        df['relative_edit_dist'] = df['edit_distance'] / df['old_text_length']
        df['is_comment_improvement'] = df['comment'].apply(lambda c: self.comment_filter(c))

        # 10% difference w.r.t old text or comment suggest meaningful improvement
        df = df[(df['relative_edit_dist'] > 0.1) | df['is_comment_improvement']]
        logger.info(f'Filtered out {df_len - df.shape[0]} rows with too small relative edit difference')
        return df

    def _filter_by_edit_distance(self, df):
        """Remove rows where the edit distance is too short"""
        df_len = df.shape[0]
        df = df[(df['edit_distance'] > self.min_edit_distance)]
        logger.info(f'Filtered out {df_len - df.shape[0]} rows with too small edit distance ({self.min_edit_distance})')
        return df

    def _filter_same_text(self, df):
        """Removes rows where new and old text is the same"""
        df_len = df.shape[0]

        def is_simple_edit(row):
            old_text = row['old_text']
            new_text = row['new_text']

            normalized_new_text = self._normalize_text(new_text)
            normalized_old_text = self._normalize_text(old_text)

            is_same_text = normalized_new_text == normalized_old_text
            if self.filter_out_simple_edits:
                return is_same_text or new_text in old_text or old_text in new_text
            else:
                return is_same_text

        mask = df.apply(is_simple_edit, axis=1)
        df = df[~mask]
        logger.info(f'Filter out {df_len - df.shape[0]} row with same text')
        return df

    def _filter_by_relative_length_difference(self, df):
        """Removes rows where relative length difference is too big"""
        df_len = df.shape[0]

        # Calculate the relative length difference based on the smaller of the two lengths
        df['old_text_length'] = df['old_text'].apply(lambda x: len(x))
        df['new_text_length'] = df['new_text'].apply(lambda x: max(1, len(x)))
        df['min_text_length'] = df[['old_text_length', 'new_text_length']].min(axis=1)
        df['relative_length_difference'] = (
            (df['new_text_length'] - df['old_text_length']) / df['min_text_length']
        ) * 100
        df['relative_length_difference'] = abs(df['relative_length_difference'])

        logger.info(f'Relative length differences: {df["relative_length_difference"].quantile([0.95, 0.99])}')
        # Remove rows where the relative difference (in %) between new and old text is too long
        # (i.e. skip very long text changes)
        max_relative_length_difference = df['relative_length_difference'].quantile(
            self.max_relative_length_difference_quantile
        )
        if max_relative_length_difference > self.maximal_max_relative_length_difference:
            max_relative_length_difference = self.maximal_max_relative_length_difference
        elif max_relative_length_difference < self.minimal_max_relative_length_difference:
            max_relative_length_difference = self.minimal_max_relative_length_difference

        df = df[(df['relative_length_difference'] <= max_relative_length_difference)]
        logger.info(
            f'Filtered out {df_len - df.shape[0]} rows with too large relative difference '
            f'(above {max_relative_length_difference}%)'
        )
        return df

    def _filter_empty_text(self, df):
        """Removes rows where text is empty"""
        df_len = df.shape[0]
        df = df[df['new_text'].str.strip() != '']
        df = df[df['old_text'].str.strip() != '']
        logger.info(f'Filter out {df_len - df.shape[0]} row with empty text')
        return df

    def _filter_short_text(self, df):
        """Removes rows where the number of words in new_text or old_text is less than 5 (often meaningless text)"""
        df_len = df.shape[0]

        df['new_text_word_count'] = df['new_text'].str.split().apply(len)
        df['old_text_word_count'] = df['old_text'].str.split().apply(len)

        df = df[
            (df['new_text_word_count'] >= self.min_words_count) & (df['old_text_word_count'] >= self.min_words_count)
        ]

        logger.info(f'Filter out {df_len - df.shape[0]} rows with less than 5 words in new_text or old_text')

        # Drop the temporary word count columns
        df = df.drop(columns=['new_text_word_count', 'old_text_word_count'])

        return df

    def _normalize_text(self, text):
        return self.whitespace_re.sub(' ', text).strip()

    @staticmethod
    def _calculate_edit_distance(row):
        old_text = row['old_text']
        new_text = row['new_text']
        return levenshtein_distance(old_text, new_text)


class WikiMarkupFilter:
    """Filter that removes diffs if text has any Wiki markup leftovers

    The filter checks for:
    * templates: {{, }}
    * link: [[, ]]
    * refs: <ref>
    * template args: |arg = value
    """

    def __init__(self):
        """Initialize filter instance"""
        # pattern: optional_whitespace pipe| any_chars = any_chars\n
        self.template_args_pattern = re.compile(r'^\s*\|.*?=.*$', re.MULTILINE)
        self.html_tags = [
            'p',
            'div',
            'li',
            'ul',
            'h1',
            'h2',
            'h3',
            'h4',
            'h5',
            'table',
            'td',
            'tr',
            'th',
            'url',
            'nowiki',
            'blockquote',
            'font',
        ]

    def filter_diffs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes rows that contain Wiki markup leftovers"""
        df_len = df.shape[0]

        mask = df['new_text'].apply(self._any_markup_remaining) | df['old_text'].apply(self._any_markup_remaining)
        df = df[~mask]

        mask = df['new_text'].apply(self._any_meta_text_remaining) | df['old_text'].apply(self._any_meta_text_remaining)
        df = df[~mask]

        logger.info(f'Filter out {df_len - df.shape[0]} rows with Wiki markup leftovers')
        return df

    def _any_markup_remaining(self, text: str) -> bool:
        if pd.isna(text):
            return False

        if '{{' in text or '}}' in text:
            logger.warning(f'Found remaining template in: {text}')
            return True
        if '[[' in text or ']]' in text:
            logger.warning(f'Found remaining link in: {text}')
            return True
        if '<ref' in text or '</ref' in text or 'ref>' in text:
            logger.warning(f'Found remaining <ref> in: {text}')
            return True
        if '<!--' in text or '-->' in text:
            logger.warning(f'Found remaining comment in: {text}')
            return True

        for tag in self.html_tags:
            if f'<{tag} ' in text or f'</{tag}>' in text or f'</{tag} ' in text:
                logger.warning(f'Found remaining HTML <{tag}> in: {text}')
                return True

        return bool(self.template_args_pattern.search(text))

    def _any_meta_text_remaining(self, text: str) -> bool:
        """Check if there is any meta-text (like 'redirects here') in the given text"""
        if pd.isna(text):
            return False

        return 'redirects here' in text.lower()


class BleuFilter:
    """Filter that removes diffs that have low BLEU score between old and new text (0.01 percentile)

    Calculating BLEU (Bilingual Evaluation Understudy) score between old_text and new_text allows to quantify
    how similar the new_text is to the old_text, indicating whether they are likely to be related.
    If the BLEU score between old_text and new_text is very low, it suggests that the texts are not similar
    and likely belong to different parts of the Wikipedia article
    """

    def __init__(self, threshold_getter: Callable[[pd.Series], float], n_processes: int = os.cpu_count()):
        """Initialize the BLEU filter"""
        self.tokenizer = Tokenizer()
        self.threshold_getter = threshold_getter
        self.n_processes = n_processes

    def filter_diffs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out diffs which BLEU score between old and new text is below 0.01 percentile"""
        df_len = df.shape[0]

        logger.info('Calculating BLEU score')
        df = _parallelize_dataframe(df, self.n_processes, self._apply_bleu_score)
        logger.info('Finished calculating BLEU score')

        logger.warning(f'Original length: {df_len}')
        logger.warning(f"0.15 BLEU THRESHOLD: {df[df['bleu_score'] > 0.15].shape[0]}")
        t = df['bleu_score'].quantile(0.01)
        logger.warning(f"0.01 quantile BLEU THRESHOLD: {df[df['bleu_score'] > t].shape[0]}")
        t = df['bleu_score'].quantile(0.1)
        logger.warning(f"0.10 quantile BLEU THRESHOLD: {df[df['bleu_score'] > t].shape[0]}")
        logger.warning(f"BLEU PERCENTILES:\n {df['bleu_score'].quantile([0.01, 0.1, 0.2, 0.5])}")

        threshold = self.threshold_getter(df['bleu_score'])
        logger.info(f'min BLEU threshold is {threshold}')

        df = df[(threshold < df['bleu_score']) & (df['bleu_score'] < 1)]
        logger.info(f'Filtered out {df_len - df.shape[0]} rows with too small BLEU score')

        return df

    # Function to apply calculate_bleu_score to a dataframe split
    def _apply_bleu_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df['bleu_score'] = df.apply(self._calculate_bleu_score, axis=1)
        return df

    def _calculate_bleu_score(self, row) -> int:
        old_text = row['old_text']
        new_text = row['new_text']

        try:
            old_doc = self.tokenizer.tokenize(old_text)
            new_doc = self.tokenizer.tokenize(new_text)
        except (TooLongSectionException, ProblematicTokenizationPatternException) as e:
            logger.warning(
                f'Failed to tokenize text for {row["page_title"]}-{row["section"]} revision {row["rev_id"]}: {e}'
            )
            return 0

        # Tokenize texts using spaCy
        reference = [[token.text for token in old_doc]]
        candidate = [token.text for token in new_doc]

        # Calculate BLEU score
        score = sentence_bleu(reference, candidate)
        return score


class RemovalFilter:
    """Filter that removes edits that remove too much text (i.e. >1 sentence and >20% of tokens)"""

    def __init__(self, n_processes: int = os.cpu_count()):
        """Initialize filter instance"""
        self.tokenizer = Tokenizer()
        self.max_tokens_removed = 0.2
        self.n_processes = n_processes

    def filter_diffs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes rows that remove too much text"""
        logger.info('Checking diffs that remove too much text')
        df = _parallelize_dataframe(df, self.n_processes, self._apply_is_removal)
        logger.info('Finished checking diffs that remove too much text')

        df = df[~df['is_removal']]
        df.drop(columns=['is_removal'], axis=1, inplace=True)

        return df

    def _apply_is_removal(self, df: pd.DataFrame) -> pd.DataFrame:
        df['is_removal'] = df.apply(self._is_removal, axis=1)
        return df

    def _is_removal(self, row) -> int:
        old_text = row['old_text']
        new_text = row['new_text']

        try:
            old_doc = self.tokenizer.tokenize(old_text)
            new_doc = self.tokenizer.tokenize(new_text)
        except (TooLongSectionException, ProblematicTokenizationPatternException) as e:
            logger.warning(
                f'Failed to tokenize text for {row["page_title"]}-{row["section"]} revision {row["rev_id"]}: {e}'
            )
            return True

        old_text_sentences = [sent.text for sent in old_doc.sents]
        new_text_sentences = [sent.text for sent in new_doc.sents]
        old_text_sentences_len = len(old_text_sentences)
        new_text_sentences_len = len(new_text_sentences)

        old_text_tokens_len = len(old_doc)
        new_text_tokens_len = len(new_doc)

        if new_text_sentences_len >= old_text_sentences_len:
            logger.debug(
                f'Revision {row["rev_id"]} adds sentences ({new_text_sentences_len} vs {old_text_sentences_len})'
            )
            removal = False
        elif new_text_tokens_len >= old_text_tokens_len:
            logger.debug(f'Revision {row["rev_id"]} adds tokens ({new_text_tokens_len} vs {old_text_tokens_len})')
            removal = False
        elif (old_text_tokens_len - new_text_tokens_len) / old_text_tokens_len <= self.max_tokens_removed:
            logger.debug(
                f'Revision {row["rev_id"]} removes '
                f'{(old_text_tokens_len - new_text_tokens_len) / old_text_tokens_len}% tokens (allowed %)'
            )
            removal = False
        else:
            removal = True

        return removal


class SingleEditFilter:
    """Filter that keeps only rows that are unique edits per revision

    The edits are grouped by (page id, revision id), and
    if there's more than 1 of such pair the corresponding rows are dropped
    """

    def __init__(self):
        """Initialize filter instance"""
        pass

    def filter_diffs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes rows are not unique edits per revision"""
        df_len = df.shape[0]

        edits_per_rev = df.groupby(['page_id', 'rev_id']).size().reset_index(name='counts')
        single_edits_per_rev = edits_per_rev[edits_per_rev['counts'] == 1]

        df_cleaned = pd.merge(df, single_edits_per_rev, on=['page_id', 'rev_id'])
        df_cleaned.drop(['counts'], axis=1, inplace=True)

        logger.info(f'Filtered out {df_len - df_cleaned.shape[0]} rows that are not unique edits per revision')
        return df_cleaned


def _parallelize_dataframe(df: pd.DataFrame, n_processes: int, func: Callable[[pd.DataFrame], pd.DataFrame]):
    df_split = np.array_split(df, n_processes)
    with multiprocessing.Pool(n_processes) as pool:
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
    return df
