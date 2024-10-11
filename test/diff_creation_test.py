import os.path
import unittest
from pathlib import Path
from typing import Callable

from wikiprefs.diffs.diff_creation import SectionsComparator
from wikiprefs.diffs.model import SectionDiff, TextDiff
from wikiprefs.markup.markup_processing import WikiMarkupParser


class TestStringMethods(unittest.TestCase):
    resources_base = Path(__file__).parent / 'resources/diffs'

    def _test(self, file_name: str, test_section_diffs: Callable[[SectionDiff], None]) -> None:
        diffs = self._create_diffs(file_name)

        self.assertEqual(len(diffs), 1)
        s1_diff = diffs[0]
        test_section_diffs(s1_diff)

    def _create_diffs(self, file_name: str) -> [SectionDiff]:
        def read_file(v):
            with open(os.path.join(self.resources_base, v, f'{file_name}.txt')) as f:
                return f.read()

        v1_text = read_file('v1')
        v2_text = read_file('v2')

        diffs = self._create_diffs_for_text(v1_text, v2_text)
        return diffs

    def _create_diffs_for_text(self, v1_text, v2_text):
        markup_parser = WikiMarkupParser()
        comparator = SectionsComparator()

        v1 = markup_parser.parse_wiki_markup(v1_text)
        v1 = comparator.tokenize_sections(v1)
        v2 = markup_parser.parse_wiki_markup(v2_text)
        v2 = comparator.tokenize_sections(v2)

        diffs = comparator.compare_sections(v1, v2)
        return diffs

    def test_added_new_section(self):
        comparator = SectionsComparator()

        diff = comparator.compare_sections([], [('s1', 'aaa')])
        self.assertListEqual(diff, [])

    def test_removed_old_section(self):
        comparator = SectionsComparator()

        diff = comparator.compare_sections([('s1', 'aaa')], [])
        self.assertListEqual(diff, [])

    def test_only_line_differs(self):
        def test_section_diffs(section_diff: SectionDiff):
            all_segments_diffs = section_diff.all_segments_diffs
            self.assertEqual(len(all_segments_diffs), 1)
            diffs = all_segments_diffs[0].diffs

            self.assertEqual(len(diffs), 3)
            self.assertIsInstance(diffs[0], str)
            self.assertIsInstance(diffs[1], str)
            self.assertIsInstance(diffs[2], TextDiff)

            line_diff = diffs[2]
            self.assertTrue(line_diff.v1.startswith('It was developed'))
            self.assertTrue(line_diff.v2.startswith('The 7 World Trade Center was developed'))

        self._test('only_line_differs', test_section_diffs)

    def test_middle_line_differs(self):
        def test_section_diffs(section_diff: SectionDiff):
            all_segments_diffs = section_diff.all_segments_diffs
            self.assertEqual(len(all_segments_diffs), 1)
            diffs = all_segments_diffs[0].diffs

            self.assertEqual(len(diffs), 7)
            for i in [0, 1, 2, 4, 5, 6]:
                self.assertIsInstance(diffs[i], str)
            self.assertIsInstance(diffs[3], TextDiff)

            line_diff = diffs[3]

            starting_line = 'From September 8 to October 7, 2006, the work of photographer Jonathan Hyman was displayed'
            self.assertTrue(line_diff.v1.startswith(starting_line))
            self.assertTrue(line_diff.v2.startswith(starting_line))

            removed_line = 'The exhibit took place on the 45th floor while space remained available for lease.'
            self.assertTrue(removed_line in line_diff.v1)
            self.assertFalse(removed_line in line_diff.v2)

            added_line = 'The exhibit consisted of 63 photographs that captured Americans\' responses after the September 11, attacks.'
            self.assertFalse(added_line in line_diff.v1)
            self.assertTrue(added_line in line_diff.v2)

        self._test('middle_line_differs', test_section_diffs)

    def test_pov_removed(self):
        def test_section_diffs(section_diff: SectionDiff):
            all_segments_diffs = section_diff.all_segments_diffs
            self.assertEqual(len(all_segments_diffs), 1)
            diffs = all_segments_diffs[0].diffs

            self.assertEqual(len(diffs), 4)

            line_diff = diffs[3]
            self.assertTrue(line_diff.v1.startswith('However, there is in fact '))
            self.assertIsNone(line_diff.v2)

        self._test('pov_removed', test_section_diffs)

    def test_multiple_changes(self):
        def test_section_diffs(section_diff: SectionDiff):
            all_segments_diffs = section_diff.all_segments_diffs
            self.assertEqual(len(all_segments_diffs), 1)
            diffs = all_segments_diffs[0].diffs

            self.assertEqual(len(diffs), 2)

        self._test('multiple_changes', test_section_diffs)

    def test_quote_diff(self):
        test_cases = {
            'Prefix changed': [
                TextDiff(v1='Test test test:', v2='Test test change test:'),
                '{{quote|Quote quote. Quote quote}}',
                'End end'
            ],
            'Suffix changed': [
                'Test test test:',
                '{{quote|Quote quote. Quote quote}}',
                TextDiff(v1='End end', v2='End end change')
            ],
            'Quote changed': [
                'Test test test:',
                TextDiff(v1='{{quote|Quote quote. Quote quote}}', v2='{{quote|Quote quote change. Quote quote}}'),
                'End end'
            ],
            'All changed': [
                TextDiff(v1='Test test test: {{quote|Quote quote. Quote quote}} End end',
                         v2='Test test test change: {{quote|Quote quote. Quote quote change}} End change end')
            ],
            'Suffix changed (quote 2)': [
                'Test test test:',
                '{{quote|Quote quote. Quote quote}}',
                TextDiff(v1='.', v2='change.'),
                'End end'
            ],
            'All changed (quote 2)':
                [
                    TextDiff(v1='Test test test: {{quote|Quote quote. Quote quote}} . End end',
                             v2='Test test test change: {{quote|Quote quote change. Quote quote}} change. End end change')
                ]
        }

        sections_diffs = self._create_diffs('quote')
        self.assertEqual(len(sections_diffs), 6)

        diffs_dict = {d.section: d.all_segments_diffs for d in sections_diffs}
        for tc in test_cases.keys():
            expected_changes = test_cases[tc]
            with self.subTest(f'Testing {tc}'):
                all_segments_diffs = diffs_dict[tc]
                self.assertEqual(len(all_segments_diffs), 1)
                diffs = all_segments_diffs[0].diffs

                self.assertEqual(diffs, expected_changes)

    def test_list_diff(self):
        test_cases = {
            'First line changed': [TextDiff(v1='* aa', v2='* aaxx'), '\n', '* bb', '\n'],
            'First line added': [TextDiff(v1=None, v2='* xx\n'), '* aa', '\n', '* bb', ],
            'Middle line changed': ['* aa', '\n', TextDiff(v1='* bb', v2='* bbxx'), '\n', '* cc'],
            'Middle line added': ['\n', '* bb', '\n', TextDiff(v1=None, v2='* xx\n'), '* cc'],
            'Last line changed': ['\n', '* bb', '\n', TextDiff(v1='* cc', v2='* ccxx')],
            'Last line added': ['* bb', '\n', '* cc', TextDiff(v1=None, v2='\n* xx')],
            'First line removed': [TextDiff(v1='* aa\n', v2=None), '* bb', '\n', '* cc'],
            'Last line removed': ['* aa', '\n', '* bb', TextDiff(v1='\n* cc', v2=None)]
        }

        sections_diffs = self._create_diffs('list')
        self.assertEqual(len(sections_diffs), 8)

        diffs_dict = {d.section: d.all_segments_diffs for d in sections_diffs}
        for tc in test_cases.keys():
            expected_changes = test_cases[tc]
            with self.subTest(f'Testing {tc}'):
                all_segments_diffs = diffs_dict[tc]
                self.assertEqual(len(all_segments_diffs), 1)
                diffs = all_segments_diffs[0].diffs

                self.assertEqual(diffs, expected_changes)

    def test_cast_list_diff(self):
        sections_diffs = self._create_diffs('list_cast')
        all_segments_diffs = sections_diffs[0].all_segments_diffs
        self.assertEqual(len(all_segments_diffs), 1)
        diffs = all_segments_diffs[0].diffs
        self.assertEqual(len(diffs), 16)

        expected_diffs = [
            '* Sigourney Weaver as Ellen Ripley: the sole survivor of an alien attack on her ship, the "Nostromo"',
            '\n',
            TextDiff(v1='* Michael Biehn as CPL Dwayne Hicks: a corporal in the Colonial Marines',
                     v2='* Michael Biehn as Corporal Dwayne Hicks: a corporal in the Colonial Marines'),
            '\n',
            '* Paul Reiser as Carter J. Burke: a Weyland-Yutani Corporation representative',
            '\n',
            TextDiff(v1='* Lance Henriksen as Bishop: an android and the Executive Officer of the "Sulaco"',
                     v2='* Lance Henriksen as Bishop: an android aboard the "Sulaco"'),
            '\n',
            '* Carrie Henn as Rebecca "Newt" Jorden: a young girl in the colony on LV-426',
            '\n',
            TextDiff(
                v1="* Bill Paxton as PFC William Hudson: a boastful but panicky Colonial Marine private",
                v2="* Bill Paxton as Hudson: a boastful but panicky Colonial Marine private"),
            '\n',
            TextDiff(
                v1="* William Hope as Lt. Scott Gorman: the Marines' inexperienced commanding officer",
                v2="* William Hope as Gorman: the Marines' inexperienced commanding officer"),
            '\n',
            '* Ricco Ross as Frost: a private in the Colonial Marines',
            '\n',
        ]
        self.assertEqual(diffs, expected_diffs)

    def test_discography_list_diff(self):
        expected_diffs = [
            TextDiff(v1='* "Facelift" (1990) (X2 Platinum)',
                     v2='* "Facelift" (1990)'),
            '\n',
            TextDiff(v1='* "Dirt" (1992) (X6 Platinum)',
                     v2='* "Dirt" (1992)'),
            '\n',
            TextDiff(
                v1='* "Alice in Chains" (1995) (X3 Platinum)',
                v2='* "Alice in Chains" (1995)'),
            '\n',
            TextDiff(
                v1='* "Black Gives Way to Blue" (2009) (Gold)',
                v2='* "Black Gives Way to Blue" (2009)')
        ]

        sections_diffs = self._create_diffs('list_discography')
        all_segments_diffs = sections_diffs[0].all_segments_diffs
        self.assertEqual(len(all_segments_diffs), 1)
        diffs = all_segments_diffs[0].diffs
        self.assertEqual(len(diffs), len(expected_diffs))

        self.assertEqual(diffs, expected_diffs)

    def test_multiple_lists_diff(self):
        expected_diffs = [
            '\n',
            'The basic fielding statistics include:',
            '\n',
            TextDiff(
                v1='* Putouts: times the fielder tags, forces, or appeals a runner and he is called out as a result',
                v2='* Putouts: times the fielder catches a fly ball, tags or forces out a runner, or otherwise directly effects an out'),
            '\n',
            '* Assists: times a putout was recorded following the fielder touching the ball',
            '\n'
        ]

        sections_diffs = self._create_diffs('list_multiple')
        all_segments_diffs = sections_diffs[0].all_segments_diffs
        self.assertEqual(len(all_segments_diffs), 1)
        diffs = all_segments_diffs[0].diffs
        self.assertEqual(len(diffs), len(expected_diffs))

        self.assertEqual(diffs, expected_diffs)

    def test_diff_context(self):
        expected_diffs = [
            'The project stalled again until new Fox executive Lawrence Gordon advocated a sequel.',
            'Although relatively inexperienced, Cameron was given the director role based on his success directing "The Terminator".',
            'On an approximately $18.5 million budget, "Aliens" began principal photography in September 1985.',
            TextDiff(
                v1="Like its development, filming was tumultuous and rife with conflicts between Cameron and the British crew at Pinewood Studios over their work habits and Cameron's relative inexperience.",
                v2='Like its development, filming was tumultuous and rife with conflicts between Cameron and the British crew at Pinewood Studios.'),
            "James Horner composed the film's score.",
            'The difficult shoot also affected Horner, who was given little time to record the music.'
        ]

        sections_diffs = self._create_diffs('context_1')
        all_segments_diffs = sections_diffs[0].all_segments_diffs
        self.assertEqual(len(all_segments_diffs), 1)
        diffs = all_segments_diffs[0].diffs
        self.assertEqual(len(diffs), len(expected_diffs))

        self.assertEqual(diffs, expected_diffs)

    def test_multi_column_diff(self):
        sections_diffs = self._create_diffs('multi_column')
        all_segments_diffs = sections_diffs[0].all_segments_diffs
        self.assertEqual(len(all_segments_diffs), 1)
        diffs = all_segments_diffs[0].diffs
        self.assertEqual(len(diffs), 62)

        self.assertEqual(diffs[51], 'Romeo and Juliet')
        self.assertEqual(diffs[52], '\n')
        self.assertEqual(diffs[53].v1,
                         'Act I, scene 5 by William Miller\n'
                         'Act II, scene 5 by Robert Smirke\n'
                         'Act III, scene 5 by John Francis Rigaud\n'
                         'Capulet Finds Juliet Dead (Act IV, scene 5) by John Opie\n'
                         'Act V, scene 3 by James Northcote')
        self.assertEqual(diffs[53].v2,
                         'Act I, scene 5 by Anker Smith after William Miller\n'
                         'Act II, scene 5 by James Parker after Robert Smirke\n'
                         'Act III, scene 5 by James Stow after John Francis Rigaud\n'
                         'Capulet Finds Juliet Dead (Act IV, scene 5) by Jean Pierre Simon and William Blake after John Opie\n'
                         'Act V, scene 3 by James Heath after James Northcote')

    def test_removed_long_suffix(self):
        # i.e. removed last sentence and whole next paragraph
        sections_diffs = self._create_diffs('removed_long_suffix')
        all_segments_diffs = sections_diffs[0].all_segments_diffs
        self.assertEqual(len(all_segments_diffs), 1)
        diffs = all_segments_diffs[0].diffs
        self.assertEqual(len(diffs), 4)

        self.assertTrue(diffs[0].startswith('The Scottish driver said that he had planned to perform doughnuts'))
        self.assertTrue(
            diffs[1].startswith('Coulthard left Formula One after 15 years with 246 race starts and 13 wins'))
        self.assertTrue(diffs[2].startswith('Red Bull team principal Christian Horner said: "It\'s a great'))

        text_diff = diffs[3]
        self.assertTrue(text_diff.v1.startswith(
            'Coulthard continued to work for Red Bull Racing in 2009 as a testing and development consultant.'
            '\n\nJenson Button\'s Honda burst into flames in parc fermé'))
        self.assertEqual(text_diff.v2, None)

    def test_removed_paragraph(self):
        sections_diffs = self._create_diffs('removed_paragraph')
        all_segments_diffs = sections_diffs[0].all_segments_diffs
        self.assertEqual(len(all_segments_diffs), 1)
        diffs = all_segments_diffs[0].diffs
        self.assertEqual(len(diffs), 4)

        self.assertTrue(
            diffs[0].startswith('Coulthard left Formula One after 15 years with 246 race starts and 13 wins'))
        self.assertTrue(diffs[1].startswith('Red Bull team principal Christian Horner said: "It\'s a great'))
        self.assertTrue(diffs[2].startswith('Coulthard continued to work for Red Bull'))

        text_diff = diffs[3]
        self.assertTrue(text_diff.v1.startswith('\n\nJenson Button\'s Honda burst into flames'))
        self.assertEqual(text_diff.v2, None)

    def test_replace_last_sentence_and_remove_next_paragraph(self):
        sections_diffs = self._create_diffs('modified_long_suffix')
        all_segments_diffs = sections_diffs[0].all_segments_diffs
        self.assertEqual(len(all_segments_diffs), 1)
        diffs = all_segments_diffs[0].diffs
        self.assertEqual(len(diffs), 4)

        self.assertTrue(diffs[0].startswith('The Scottish driver said that he had planned to perform doughnuts'))
        self.assertTrue(
            diffs[1].startswith('Coulthard left Formula One after 15 years with 246 race starts and 13 wins'))
        self.assertTrue(diffs[2].startswith('Red Bull team principal Christian Horner said: "It\'s a great'))

        text_diff = diffs[3]
        self.assertTrue(text_diff.v1.startswith(
            'Coulthard continued to work for Red Bull Racing in 2009 as a testing and development consultant.'
            '\n\nJenson Button\'s Honda burst into flames in parc fermé'))
        self.assertEqual(text_diff.v2, 'Coulthard stopped to work for Red Bull Racing forever.')

    def test_modify_edge_paragraphs(self):
        sections_diffs = self._create_diffs('edge_paragraphs')
        all_segments_diffs = sections_diffs[0].all_segments_diffs

        self.assertEqual(len(all_segments_diffs), 2)

        segment_1 = all_segments_diffs[0]
        self.assertEqual(len(segment_1.diffs), 4)
        self.assertEqual(segment_1.diffs[0], TextDiff(v1=None, v2='Change in first paragraph.'))

        segment_2 = all_segments_diffs[1]
        self.assertEqual(len(segment_2.diffs), 4)
        self.assertEqual(segment_2.diffs[3], TextDiff(
            v1='Coulthard continued to work for Red Bull Racing in 2009 as a testing and development consultant',
            v2='Coulthard continued to work for Red Bull Racing in 2009 as a testing and development consultant. Change in last paragraph.'))

    def test_compare_same_sections(self):
        with open(os.path.join(self.resources_base, 'v1', f'edge_paragraphs.txt')) as f:
            text = f.read()

        sections_diffs = self._create_diffs_for_text(text, text)
        self.assertEqual(len(sections_diffs), 0)

    def test_compare_math(self):
        sections_diffs = self._create_diffs('math1')
        all_segments_diffs = sections_diffs[0].all_segments_diffs

        self.assertEqual(len(all_segments_diffs), 1)

        segment_1 = all_segments_diffs[0]
        self.assertEqual(len(segment_1.diffs), 11)

        diffs = segment_1.diffs
        self.assertEqual(diffs[2],
                         TextDiff(
                             v1=' <math> \n C_1 = \\mathbf{D} \\cdot \\mathbf{D} + \\mathbf{L} \\cdot \\mathbf{L} = \\frac{mk^2}{2|E|}, \n </math>',
                             v2="<math> \\begin{align} \n C_1 &= \\mathbf{D} \\cdot \\mathbf{D} + \\mathbf{L} \\cdot \\mathbf{L} = \\frac{mk^2}{2|E|}, \\\\ \n C_2 &= \\mathbf{D} \\cdot \\mathbf{L} = 0, \n \\end{align}</math>\n\nand have vanishing Poisson brackets with all components of {{math|'''D'''}} and {{math|'''L'''}} ,")
                         )
        self.assertEqual(diffs[8],
                         TextDiff(
                             v1='However, the other invariant, "C"1, is non-trivial and depends only on "m", "k" and "E".  ',
                             v2='However, the other invariant, "C"1, is non-trivial and depends only on {{mvar|m}}, {{mvar|k}} and {{mvar|E}}.')
                         )

    def test_compare_math_2(self):
        sections_diffs = self._create_diffs('math2')
        all_segments_diffs = sections_diffs[0].all_segments_diffs

        self.assertEqual(len(all_segments_diffs), 1)

        segment_1 = all_segments_diffs[0]
        self.assertEqual(len(segment_1.diffs), 13)

        diffs = segment_1.diffs
        self.assertEqual(diffs[0],
                         TextDiff(
                             v1="The constancy of the LRL vector can also be derived from the Hamilton–Jacobi equation in parabolic coordinates {{nowrap|1=(''ξ'', ''η'')}}, which are defined by the equations\n\n <math> \n \\xi = r + x, \n </math>",
                             v2="The constancy of the LRL vector can also be derived from the Hamilton–Jacobi equation in parabolic coordinates {{math|1=(''ξ'', ''η'')}} , which are defined by the equations")
                         )
        self.assertEqual(diffs[2],
                         TextDiff(
                             v1=' <math> \n \\eta = r - x, \n </math>\n\nwhere "r" represents the radius in the plane of the orbit\n\n <math> \n r = \\sqrt{x^2 + y^2}. \n </math>\n\nThe inversion of these coordinates is\n\n <math> \n x = \\frac{1}{2} (\\xi - \\eta), \n </math>',
                             v2='<math>\\begin{align} \n \\xi &= r + x, \\\\ \n \\eta &= r - x, \n \\end{align}</math>')
                         )
        self.assertEqual(diffs[12],
                         TextDiff(
                             v1='where Γ is a constant of motion.  Subtraction and re-expression in terms of the Cartesian momenta "p"x and "p"y shows that Γ is equivalent to the LRL vector\n\n <math> \n \\Gamma = p_y (x p_y - y p_x) - mk\\frac{x}{r} = A_x. \n </math>',
                             v2='<math> \n \\Gamma = p_y (x p_y - y p_x) - mk\\frac{x}{r} = A_x. \n </math>')
                         )

    def test_compare_math_3(self):
        self.skipTest('FIXME: handle broken markup')

        sections_diffs = self._create_diffs('math3')
        all_segments_diffs = sections_diffs[0].all_segments_diffs

    def test_compare_chess_moves(self):
        # FIXME: chess annotation symbols are not fully correctly tokenized into sentences by Spacy
        # but there's not syntax we could use for detecting that it's a special notation, e.g.:
        # (...) arises after 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 6.Bg5 e6 7.f4 Qb6. This has long been ...
        sections_diffs = self._create_diffs('chess')
        all_segments_diffs = sections_diffs[0].all_segments_diffs
        self.assertEqual(len(all_segments_diffs), 1)
        diffs = all_segments_diffs[0].diffs
        self.assertEqual(len(diffs), 7)


if __name__ == '__main__':
    unittest.main()
