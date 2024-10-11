import operator
import unittest

import pandas as pd
from pandas._testing import assert_frame_equal

from wikiprefs.collect_retained_diffs import DiffRow
from wikiprefs.diffs.filtering import OutliersFilter, RemovalFilter, WikiMarkupFilter, BleuFilter, SingleEditFilter


def _create_diff(new_text, old_text):
    row = DiffRow(
        page_title=f'test_page',
        page_id=1,
        rev_id=2,
        prev_rev_id=1,
        timestamp='5.5.2005',
        contributor='x',
        comment='removed text',
        section='intro',
        old_text=old_text,
        new_text=new_text,
    )
    return row


def _create_text_with_template_arg(arg_str):
    text_with_template_args = f'''
    Opening text
    {arg_str}
    Closing text
    '''
    return text_with_template_args


class TestDatasetFiltering(unittest.TestCase):

    def test_text_filtering(self):
        diffs = []
        row = _create_diff('', 'old')
        diffs.append(row)
        row = _create_diff('new', '')
        diffs.append(row)
        row = _create_diff('same same same same same', 'same same same same same')
        diffs.append(row)
        row = _create_diff('a b b b b', 'b b b b b')  # filtered out by edit distance check
        diffs.append(row)
        df = pd.DataFrame([p.__dict__ for p in diffs])

        filter = OutliersFilter(lambda _: True, 0.99, False)
        df = filter.filter_diffs(df)

        self.assertEqual(len(df), 0)

    def test_relative_length_difference_filtering(self):
        diffs = []
        for i in range(1, 11):
            row = DiffRow(
                page_title=f'test_{i}',
                page_id=i,
                rev_id=2,
                prev_rev_id=1,
                timestamp='5.5.2005',
                contributor='x',
                comment='removed text',
                section='intro',
                old_text='a a a a a' * i,  # old text is longer
                new_text='b b b b b' * max(1, i - 2)
            )
            diffs.append(row)
        df = pd.DataFrame([p.__dict__ for p in diffs])

        filter = OutliersFilter(lambda _: True, 0.99, False)
        df = filter.filter_diffs(df)

        # 1 for relative length difference
        self.assertEqual(len(df), 9)

    def test_markup_filtering(self):
        diffs = []
        # correct text
        diffs.append(_create_diff('new', 'old'))
        # templates, links, refs
        diffs.append(_create_diff('new', 'old {{unclosed template'))
        diffs.append(_create_diff('new unopened template}} more text', 'old'))
        diffs.append(_create_diff('new unopened link]]', 'old'))
        diffs.append(_create_diff('new', 'old unclosed link]]'))
        diffs.append(_create_diff('new', 'old </ref>'))
        diffs.append(_create_diff('new <ref', 'old'))
        # template args
        diffs.append(_create_diff('new text', _create_text_with_template_arg('| Recorded    = Sep 1991-Dec 1991')))
        diffs.append(_create_diff('new text', _create_text_with_template_arg('| Reviews     = ')))
        diffs.append(_create_diff('new text', _create_text_with_template_arg(' | Length      = 52:37')))
        diffs.append(_create_diff('new text',
                                  'A cladogram by Tortosa \"et al.\"2013 places \"Majungasaurus\" in a new '
                                  'subfamily, Majungasaurinae.\n\n{{clade| style=font-size:100%; line-height:100%\n'
                                  '|label1=Abelisauridae\n|1='))
        # meta-text
        diffs.append(_create_diff('another article redirects here. New text', 'old text'))
        diffs.append(_create_diff('new text', 'REDIRECTS HERE old text'))
        # create dataframe
        df = pd.DataFrame([p.__dict__ for p in diffs])

        filter = WikiMarkupFilter()
        df = filter.filter_diffs(df)

        self.assertEqual(len(df), 1)

    def test_bleu_filter(self):
        # given
        data = {
            'old_text': [
                "The quick brown fox jumps over the lazy dog.",
                "An apple a day keeps the doctor away.",
                "She sells sea shells by the sea shore."
            ],
            'new_text': [
                "A quick brown fox jumped over lazy dogs.",
                "Eating apples daily can keep your doctor visits rare.",
                "She sells shells by the shore."
            ],
            'page_title': ["Test Page 1", "Test Page 2", "Test Page 3"],
            'rev_id': [101, 102, 103]
        }
        df = pd.DataFrame(data)

        # when
        filter = BleuFilter(get_threshold, n_processes=2)
        filtered_df = filter.filter_diffs(df)

        # then
        self.assertEqual(len(filtered_df), 2)  # Check if the correct number of rows are filtered out
        self.assertTrue((filtered_df['bleu_score'] > 8.58 - 80).all())  # Check if filtering condition is met

    def test_single_edit_filter(self):
        # given
        data = {
            'page_title': ['A', 'A', 'B', 'B', 'C'],
            'page_id': [1, 1, 2, 2, 3],
            'rev_id': [101, 101, 102, 103, 104],
            'other_data': ['text1', 'text2', 'text3', 'text4', 'text5']
        }
        df = pd.DataFrame(data)

        # when
        filter_instance = SingleEditFilter()
        result_df = filter_instance.filter_diffs(df)

        # then
        expected_data = {
            'page_title': ['B', 'B', 'C'],
            'page_id': [2, 2, 3],
            'rev_id': [102, 103, 104],
            'other_data': ['text3', 'text4', 'text5']
        }
        expected_df = pd.DataFrame(expected_data)
        assert_frame_equal(result_df, expected_df)

    def test_text_removal_filter(self):
        diffs = []
        diffs.append(_create_diff('Sentence 1.', 'Sentence 1'))
        diffs.append(_create_diff('Sentence 1.', 'Sentence 1. Sentence 2'))
        diffs.append(_create_diff('Sentence 1 but longer', 'Sentence 1. Sentence 2'))
        diffs.append(_create_diff('Sentence 1. Sentence 2', 'Sentence 1 a b c d e f g h. Sentence 2'))
        diffs.append(_create_diff('Sentence 1 a b c d e f g h', 'Sentence 1 a b c d e f g h. Sentence 2'))
        df = pd.DataFrame(diffs)

        # when
        filter_instance = RemovalFilter(n_processes=2)
        result_df = filter_instance.filter_diffs(df)
        result_df.reset_index(drop=True, inplace=True)

        # then
        expected_diffs = operator.itemgetter(*[0, 2, 4])(diffs)
        expected_df = pd.DataFrame(expected_diffs)
        assert_frame_equal(result_df, expected_df)


def get_threshold(bleu_score: pd.Series) -> float:
    return bleu_score.quantile(0.01)


if __name__ == '__main__':
    unittest.main()
