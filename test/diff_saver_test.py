import unittest
from csv import DictWriter
from unittest.mock import Mock, call, patch

from wikiprefs.collect_retained_diffs import DiffRow, DiffSaver
from wikiprefs.diffs.model import TextDiff
from wikiprefs.utils.xml_utils import Revision


def _create_call(old_text: str, new_text: str) -> call:
    c = call.writerow({
        'page_title': 'page',
        'page_id': 1,
        'rev_id': 2,
        'prev_rev_id': 1,
        'timestamp': '2002-12-11T09:39:56Z',
        'contributor': 'user',
        'comment': 'comment',
        'section': 'Intro section',
        'old_text': old_text,
        'new_text': new_text,
    }),
    c = c[0]
    return c


class TestDiffSaver(unittest.TestCase):
    def setUp(self):
        self.writer_mock = Mock(spec=DictWriter)
        self.base_row = DiffRow(
            page_title='page',
            page_id=1,
            rev_id=2,
            prev_rev_id=1,
            timestamp='2002-12-11T09:39:56Z',
            contributor='user',
            comment='comment',
            section='Intro section'
        )
        self.revision = Revision(
            id='2',
            parent_id='1',
            timestamp='2002-12-11T09:39:56Z',
            is_minor=False,
            contributor='user',
            comment='comment'
        )

    def test_para_break(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_text('text')
        self.diff_saver.next_text('\n\n')

        self.assertEqual(self.diff_saver.curr_plain_text, [])
        self.writer_mock.writerow.assert_not_called()

    def test_text_with_para_break(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        with patch.object(self.diff_saver, '_save_diff_and_reset',
                          wraps=self.diff_saver._save_diff_and_reset) as mock_save_diff:
            self.diff_saver.next_text('hello\n\nworld')

            mock_save_diff.assert_called_once()
            self.writer_mock.writerow.assert_not_called()
            self.assertEqual(self.diff_saver.curr_plain_text, ['world'])

    def test_diff_not_retained(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        with patch.object(self.diff_saver, '_save_diff_and_reset',
                          wraps=self.diff_saver._save_diff_and_reset) as mock_save_diff:
            self.diff_saver.next_text('text')
            self.diff_saver.next_diff(TextDiff('old', 'new'), is_retained=False)

            mock_save_diff.assert_called_once()
            self.writer_mock.writerow.assert_not_called()
            # Assert reset
            self.assertEqual(self.diff_saver.curr_diffs_seq.old_text, '')
            self.assertEqual(self.diff_saver.curr_diffs_seq.new_text, '')

    def test_text_diff_text_removed__fa_criteris(self):
        self.revision.comment = 'refactoring article [[WP:FACRITERIA]]'
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, True)

        try:
            self.diff_saver.next_text('text text ')
            self.diff_saver.next_diff(TextDiff('old\n\n', None), is_retained=True)

            self.writer_mock.writerow.assert_called_once()
            self.assertEqual(self.diff_saver.base_row.old_text, 'text text old')
            self.assertEqual(self.diff_saver.base_row.new_text, 'text text')
        finally:
            self.revision.comment = 'comment'

    def test_text_diff_text_removed(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_text('text ')
        self.diff_saver.next_diff(TextDiff('old\n\n', None), is_retained=True)

        self.writer_mock.writerow.assert_not_called()
        # Assert reset
        self.assertEqual(self.diff_saver.curr_diffs_seq.old_text, '')
        self.assertEqual(self.diff_saver.curr_diffs_seq.new_text, '')

    def test_next_diff_too_npov_long_text_removed(self):
        self.revision.comment = 'removing POV'
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, True)
        try:
            with patch.object(self.diff_saver, '_save_diff_and_reset',
                              wraps=self.diff_saver._save_diff_and_reset) as mock_save_diff:
                self.diff_saver.next_text('text ')
                self.diff_saver.next_diff(TextDiff('old' * 10, None), is_retained=True)
                self.diff_saver.next_text('suffix\n\n')

                mock_save_diff.assert_called_once()
                self.writer_mock.writerow.assert_not_called()
                # Assert reset
                self.assertEqual(self.diff_saver.curr_plain_text, [])
                self.writer_mock.writerow.assert_not_called()
        finally:
            self.revision.comment = 'comment'

    def test_next_diff_text_only_added(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        with patch.object(self.diff_saver, '_save_diff_and_reset',
                          wraps=self.diff_saver._save_diff_and_reset) as mock_save_diff:
            self.diff_saver.next_diff(TextDiff(None, 'added text '), is_retained=True)
            self.diff_saver.next_text('suffix\n\n')

            mock_save_diff.assert_called_once()
            self.writer_mock.writerow.assert_called_once()
            self.assertEqual(self.diff_saver.base_row.old_text, 'suffix')
            self.assertEqual(self.diff_saver.base_row.new_text, 'added text suffix')

    def test_text_diff_text_invalid_diff(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_text('prefix 1 ')
        self.diff_saver.next_diff(TextDiff('old 1 ', 'new 1 '), is_retained=True)
        self.diff_saver.next_text('prefix 2 ')

        self.assertEqual(self.diff_saver.curr_diffs_seq.old_text, 'prefix 1 old 1 ')
        self.assertEqual(self.diff_saver.curr_diffs_seq.new_text, 'prefix 1 new 1 ')
        self.assertEqual(self.diff_saver.curr_plain_text, ['prefix 2 '])

        # flush diff
        self.diff_saver.next_diff(TextDiff('old 2 ', 'new 2 '), is_retained=False)
        # assert curr diffs are reset
        self.assertEqual(self.diff_saver.curr_plain_text, [])
        self.assertEqual(self.diff_saver.curr_diffs_seq.old_text, '')
        self.assertEqual(self.diff_saver.curr_diffs_seq.new_text, '')
        # assert previous diffs were saved
        self.writer_mock.writerow.assert_called_once()
        self.assertEqual(self.diff_saver.base_row.old_text, 'prefix 1 old 1')
        self.assertEqual(self.diff_saver.base_row.new_text, 'prefix 1 new 1')

    def test_text_diff_text_diff_break(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_text('prefix 1 ')
        self.diff_saver.next_diff(TextDiff('old 1 ', 'new 1 '), is_retained=True)
        self.diff_saver.next_text('prefix 2 ')
        self.diff_saver.next_diff(TextDiff('old 2', 'new 2'), is_retained=True)

        self.assertEqual(self.diff_saver.curr_diffs_seq.old_text, 'prefix 1 old 1 prefix 2 old 2')
        self.assertEqual(self.diff_saver.curr_diffs_seq.new_text, 'prefix 1 new 1 prefix 2 new 2')
        self.assertEqual(self.diff_saver.curr_plain_text, [])

        # flush diff
        self.diff_saver.next_text('\n\n')
        # assert curr diffs are reset
        self.assertEqual(self.diff_saver.curr_plain_text, [])
        self.assertEqual(self.diff_saver.curr_diffs_seq.old_text, '')
        self.assertEqual(self.diff_saver.curr_diffs_seq.new_text, '')
        # assert previous diffs were saved
        self.writer_mock.writerow.assert_called_once()
        self.assertEqual(self.diff_saver.base_row.old_text, 'prefix 1 old 1 prefix 2 old 2')
        self.assertEqual(self.diff_saver.base_row.new_text, 'prefix 1 new 1 prefix 2 new 2')

    def test_prefix_removed(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_diff(TextDiff('old 1 ', ''), is_retained=True)
        self.diff_saver.next_text('text')
        self.diff_saver.next_text('\n\n')

        self.writer_mock.writerow.assert_not_called()
        # assert curr diffs are reset
        self.assertEqual(self.diff_saver.curr_plain_text, [])
        self.assertEqual(self.diff_saver.curr_diffs_seq.old_text, '')
        self.assertEqual(self.diff_saver.curr_diffs_seq.new_text, '')

    def test_prefix_removed_then_diff(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_diff(TextDiff('old prefix ', ''), is_retained=True)
        self.diff_saver.next_text('text ')
        self.diff_saver.next_diff(TextDiff('old text', 'new text'), is_retained=True)
        # flush
        self.diff_saver.next_text('\n\n')
        # assert diffs were saved
        self.writer_mock.writerow.assert_called_once()
        self.assertEqual(self.diff_saver.base_row.old_text, 'old prefix text old text')
        self.assertEqual(self.diff_saver.base_row.new_text, 'text new text')

    def test_prefix_moved_to_suffix(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_diff(TextDiff('old prefix ', ''), is_retained=True)
        self.diff_saver.next_text('text ')
        self.diff_saver.next_diff(TextDiff('', 'old prefix'), is_retained=True)
        # flush
        self.diff_saver.next_text('\n\n')
        # assert diffs were saved
        self.writer_mock.writerow.assert_called_once()
        self.assertEqual(self.diff_saver.base_row.old_text, 'old prefix text')
        self.assertEqual(self.diff_saver.base_row.new_text, 'text old prefix')

    def test_replace_last_sentence_and_remove_next_paragraph(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_text('text ')
        self.diff_saver.next_diff(TextDiff('Last sentence.\n\nNext paragraph', 'Modified last sentence'),
                                  is_retained=True)

        # paragraph break in old text should trigger a flush
        self.writer_mock.writerow.assert_called_once()
        self.assertEqual(self.diff_saver.base_row.old_text, 'text Last sentence.')
        self.assertEqual(self.diff_saver.base_row.new_text, 'text Modified last sentence')

    def test_removed_last_sentence_and_next_paragraph(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        with patch.object(self.diff_saver, '_save_diff_and_reset',
                          wraps=self.diff_saver._save_diff_and_reset) as mock_save_diff:
            self.diff_saver.next_text('text ')
            self.diff_saver.next_diff(TextDiff('Last sentence.\n\nNext paragraph', None),
                                      is_retained=True)

            # paragraph break in old text should trigger a flush
            mock_save_diff.assert_called_once()
            self.writer_mock.writerow.assert_not_called()
            # assert curr diffs are reset
            self.assertEqual(self.diff_saver.curr_plain_text, [])
            self.assertEqual(self.diff_saver.curr_diffs_seq.old_text, '')
            self.assertEqual(self.diff_saver.curr_diffs_seq.new_text, '')

    def test_diff_then_removed_last_sentence_and_next_paragraph(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_text('Intro ')
        self.diff_saver.next_diff(TextDiff('Old text ', 'New text '),
                                  is_retained=True)
        self.diff_saver.next_text('Common text ')
        self.diff_saver.next_diff(TextDiff('Last sentence.\n\nNext paragraph', None),
                                  is_retained=True)

        # paragraph break in old text should trigger a flush
        self.writer_mock.writerow.assert_called_once()
        self.assertEqual(self.diff_saver.base_row.old_text, 'Intro Old text Common text Last sentence.')
        self.assertEqual(self.diff_saver.base_row.new_text, 'Intro New text Common text')

    def test_removed_next_paragraph(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_text('text')
        self.diff_saver.next_diff(TextDiff('\n\nNext paragraph', None), is_retained=True)

        self.diff_saver.finish()
        self.writer_mock.writerow.assert_not_called()

    def test_save_diff_and_reset_saves_correctly(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.curr_diffs_seq.old_text = 'old'
        self.diff_saver.curr_diffs_seq.new_text = 'new'

        self.diff_saver._save_diff_and_reset()

        self.writer_mock.writerow.assert_called_once()
        self.assertEqual(self.diff_saver.diffs_count, 1)

    def test_save_diff_and_reset_with_empty_texts(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.curr_diffs_seq.old_text = 'old'
        self.diff_saver.curr_diffs_seq.new_text = ''
        self.diff_saver.next_text('text')

        self.diff_saver._save_diff_and_reset()

        self.assertEqual(self.diff_saver.curr_plain_text, [])
        self.writer_mock.writerow.assert_not_called()

    def test_list(self):
        self.diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)

        self.diff_saver.next_text('* l1\n')
        self.diff_saver.next_text('* l2\n')
        self.diff_saver.next_diff(TextDiff('* old 1\n', '* new 1\n'), is_retained=True)
        self.diff_saver.finish()

        self.writer_mock.writerow.assert_called_once()

    def test_math_in_new_paragraph(self):
        diffs = [
            "While some of Euler's proofs are not acceptable by modern standards of mathematical rigour (in particular his reliance on the principle of the generality of algebra), his ideas led to many great advances.",
            '\n',
            'Euler is well known in analysis for his frequent use and development of power series, the expression of functions as sums of infinitely many terms, such as',
            TextDiff(
                v1='\n\n <math>e^x = \\sum_{n=0}^\\infty {x^n \\over n!} = \\lim_{n \\to \\infty} \\left(\\frac{1}{0!} + \\frac{x}{1!} + \\frac{x^2}{2!} + \\cdots + \\frac{x^n}{n!}\\right).</math>',
                v2='\n<math>e^x = \\sum_{n=0}^\\infty {x^n \\over n!} = \\lim_{n \\to \\infty} \\left(\\frac{1}{0!} + \\frac{x}{1!} + \\frac{x^2}{2!} + \\cdots + \\frac{x^n}{n!}\\right).</math>'),
            '\n\n',
            'Euler directly proved the power series expansions for', "{{math|''e''}}",
            'and the inverse tangent function.',
            '(Indirect proof via the inverse power series technique was given by Newton and Leibniz between 1670 and 1680.)',
            'His daring use of power series enabled him to solve the famous Basel problem in 1735 (he provided a more elaborate argument in 1741):',
            TextDiff(
                v1='\n\n <math>\\sum_{n=1}^\\infty {1 \\over n^2} = \\lim_{n \\to \\infty}\\left(\\frac{1}{1^2} + \\frac{1}{2^2} + \\frac{1}{3^2} + \\cdots + \\frac{1}{n^2}\\right) = \\frac{\\pi ^2}{6}.</math>',
                v2='\n<math>\\sum_{n=1}^\\infty {1 \\over n^2} = \\lim_{n \\to \\infty}\\left(\\frac{1}{1^2} + \\frac{1}{2^2} + \\frac{1}{3^2} + \\cdots + \\frac{1}{n^2}\\right) = \\frac{\\pi ^2}{6}.</math>'),
            '\n\n',
            'Euler introduced the use of the exponential function and logarithms in analytic proofs.',
            'He discovered ways to express various logarithmic functions using power series, and he successfully defined logarithms for negative and complex numbers, thus greatly expanding the scope of mathematical applications of logarithms.'
        ]

        diff_saver = DiffSaver(self.writer_mock, self.base_row, self.revision, False)
        for d in diffs:
            if isinstance(d, str):
                diff_saver.next_text(d)
            else:
                diff_saver.next_diff(d, True)
        diff_saver.finish()

        print(self.writer_mock.mock_calls)

        calls = [
            _create_call(
                "While some of Euler's proofs are not acceptable by modern standards of mathematical rigour "
                "(in particular his reliance on the principle of the generality of algebra), his ideas led to many great advances.\n"
                "Euler is well known in analysis for his frequent use and development of power series, the expression of functions as sums of infinitely many terms, such as"
                "\n\n <math>e^x = \\sum_{n=0}^\\infty {x^n \\over n!} = \\lim_{n \\to \\infty} \\left(\\frac{1}{0!} + \\frac{x}{1!} + \\frac{x^2}{2!} + \\cdots + \\frac{x^n}{n!}\\right).</math>",
                "While some of Euler's proofs are not acceptable by modern standards of mathematical rigour "
                "(in particular his reliance on the principle of the generality of algebra), his ideas led to many great advances.\n"
                "Euler is well known in analysis for his frequent use and development of power series, the expression of functions as sums of infinitely many terms, such as"
                "\n<math>e^x = \\sum_{n=0}^\\infty {x^n \\over n!} = \\lim_{n \\to \\infty} \\left(\\frac{1}{0!} + \\frac{x}{1!} + \\frac{x^2}{2!} + \\cdots + \\frac{x^n}{n!}\\right).</math>"
            ),
            _create_call(
                "Euler directly proved the power series expansions for{{math|''e''}}and the inverse tangent function."
                "(Indirect proof via the inverse power series technique was given by Newton and Leibniz between 1670 and 1680.)"
                "His daring use of power series enabled him to solve the famous Basel problem in 1735 (he provided a more elaborate argument in 1741):"
                "\n\n <math>\\sum_{n=1}^\\infty {1 \\over n^2} = \\lim_{n \\to \\infty}\\left(\\frac{1}{1^2} + \\frac{1}{2^2} + \\frac{1}{3^2} + \\cdots + \\frac{1}{n^2}\\right) = \\frac{\\pi ^2}{6}.</math>",
                "Euler directly proved the power series expansions for{{math|''e''}}and the inverse tangent function."
                "(Indirect proof via the inverse power series technique was given by Newton and Leibniz between 1670 and 1680.)"
                "His daring use of power series enabled him to solve the famous Basel problem in 1735 (he provided a more elaborate argument in 1741):"
                "\n<math>\\sum_{n=1}^\\infty {1 \\over n^2} = \\lim_{n \\to \\infty}\\left(\\frac{1}{1^2} + \\frac{1}{2^2} + \\frac{1}{3^2} + \\cdots + \\frac{1}{n^2}\\right) = \\frac{\\pi ^2}{6}.</math>"
            ),
        ]
        self.writer_mock.assert_has_calls(calls, any_order=False)


if __name__ == '__main__':
    unittest.main()
