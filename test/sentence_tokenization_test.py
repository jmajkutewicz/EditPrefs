import os
import unittest
from pathlib import Path

from wikiprefs.diffs.diff_creation import SectionsComparator
from wikiprefs.diffs.tokenization import ProblematicTokenizationPatternException
from wikiprefs.markup.markup_processing import WikiMarkupParser


class TestDemonstrateSubtest(unittest.TestCase):
    resources_path = Path(__file__).parent / 'resources/tokenization'

    def test_tokenization(self):
        test_cases = [
            ('testing real quote', 'quote.txt', [
                'The world of Earthsea',
                'This includes',
                'In addition',
                'Describing this',
                'The concept of',
                'While at',
                '{{quote|But ',
                '\n\n',
                'The influence',
                'At the end',
                'He has also',
                'Light and dark',
                'Reviewers have',
                'In emphasizing',
                'This tendency',
                '\n',
            ]),
            ('testing simple quote with templates', 'quote_2.txt', [
                'Sentence 1 {{c}}',
                'Sentence {{b}} 2',
                '{{quote|Quoted sentence1. Quoted sentence2{{sfn|abc}}',
                'Sentence 3 {{a}} c',
            ]),
            ('testing block quote', 'blockquote.txt', [
                'On Christmas Eve',
                'Only ten',
                'Now, the',
                'Around them',
                '\n\n',
                '{{Blockquote  | It ',
                '\n\n',
                'Kraft again ',
                'He played a ',
                'Called into ',
            ]),
            ('testing nowrap', 'nowrap.txt', [
                'In the acute',
                'During this phase, {{nowrap|\'\'T. cruzi\'\'}} can',
                'During the initial',
                '\n',
            ]),
            ('testing list', 'list.txt', [
                '* All Elite',
                '\n',
                '**AEW World',
                '\n',
                '**AEW Dynamite ',
                '\n',
                '***Best Moment',
                '\n',
                '***Best Mic',
                '\n',
                '*The Baltimore',
                '\n',
                '**Feud',
                '\n',
                '*Cauliflower',
                '\n',
                '** Iron',
                '\n',
                '*Pro',
                '\n',
                '**Comeback',
                '\n',
                '**Feud of the',
                '\n',
                '** Feud of the Year',
                '\n',
                '** Match of the Year',
                '\n',
                '** E',
                '\n',
            ]),
            ('testing no split on tag', 'tag.txt', [
                'Due to the cancellation of her 2020 tour',
                'A:',  # FIXME: spacy incorrectly splits this sentence in the middle of a concert name
                "Saigo no Trouble Final",
                'This was without',
                'On December 2',
                'Later that same month',
                '"Countdown Live 2020-2021',
                'Hamasaki later confirmed',
                '\n',
            ]),
            ('testing <math> 1', 'math.txt', [
                'If one places <math> 0.9 </math>',
                'For any number <math>x</math> ',
                'So, it does not',
                'Meanwhile, every number',
                'Therefore, <math> 0.999\\ldots</math>',
                'Because <math>0.999\\ldots</math> cannot',
                '\n',
            ]),
            ('testing <math> 2', 'math_2.txt', [
                'A common development',
                'In general:',
                '\n',
                '<math display="block"> b_0 . b',
                '\n',
            ]),
            ('testing <math> 3', 'math_3.txt', [
                'The nested intervals ',
                'To directly exploit',
                '\n',
            ]),
            ('testing <math> 4', 'math_4.txt', [
                'The Casimir invariants for negative energies are',
                '\n',
                ' <math> \n C_1 = \\mathbf{D} \\cdot \\mathbf{D} + \\mathbf{L} \\cdot \\mathbf{L} = \\frac{mk^2}{2|E|}, \n </math>',
                '\n',
                ' <math> \n C_2 = \\mathbf{D} \\cdot \\mathbf{L} = 0, \n </math>',
                '\n\n',
                'and have vanishing Poisson brackets with all components of D and L,',
                '\n',
                ' <math> \n \\{ C_1, L_i \\} = \\{ C_1, D_i\\} = \n \\{ C_2, L_i \\} = \\{ C_2, D_i \\} = 0. \n </math>',
                '\n',
                '"C"2 is trivially zero, since the two vectors are always perpendicular.',
                '\n\n',
                'However, the other invariant, "C"1, is non-trivial and depends only on "m", "k" and "E".  ',
                'Upon canonical quantization, this invariant allows the energy levels of hydrogen-like atoms',
                'This derivation is discussed in detail in the next section.']),
            ('testing <math> 5', 'math_5.txt', [
                "The connection between the rotational symmetry described above",
                'This theorem, which is used for finding constants of motion',
                '\n\n',
                ' <math> \n \\delta q_i = \\varepsilon g_i(\\mathbf{q}, \\mathbf{\\dot{q}}, t) \n </math>',
                '\n\n',
                'that causes the Lagrangian to vary to first order by a total time derivative',
                '\n\n',
                ' <math> \n \\delta L = \\varepsilon \\frac{d}{dt} G(\\mathbf{q}, t) \n </math>',
                '\n\n',
                'corresponds to a conserved quantity Γ',
                '\n\n',
                ' <math> \n \\Gamma = -G + \\sum_i g_i \\left( \\frac{\\partial L}{\\partial \\dot{q}_i}\\right) \n </math>']),
            ('testing <math> 6', 'math_6.txt', [
                'The "shape" and "orientation" of the orbits can be determined from the LRL vector as follows.',
                'Taking the dot product of', "{{math|'''A'''}}",
                'with the position vector', "{{math|'''r'''}}",
                'gives the equation',
                '\n',
                '<math> \n \\mathbf{A} \\cdot \\mathbf{r} = A \\cdot r \\cdot \\cos\\theta = \n \\mathbf{r} \\cdot \\left( \\mathbf{p} \\times \\mathbf{L} \\right) - mkr, \n </math>',
                '\n',
                'where {{mvar|θ}} is the angle between',
                "{{math|'''r'''}}", 'and',
                "{{math|'''A'''}}",
                '(Figure 2).',
                'Permuting the scalar triple product yields',
                '\n',
                '<math> \n \\mathbf{r} \\cdot\\left(\\mathbf{p}\\times \\mathbf{L}\\right) = \n \\left(\\mathbf{r} \\times \\mathbf{p}\\right)\\cdot\\mathbf{L} = \n \\mathbf{L}\\cdot\\mathbf{L}=L^2 \n </math>']),
        ]

        sc = SectionsComparator()
        for msg, filename, expected in test_cases:
            with self.subTest(msg=msg, filename=filename, expected=expected):
                file_path = os.path.join(self.resources_path, filename)
                with open(file_path, 'r', encoding='UTF-8') as src:
                    text = src.read()

                sentences = sc._tokenize_text(text)
                self.assertEqual(len(sentences), len(expected))

                # check if number of non-whitespace characters matches
                text_len = sum(not chr.isspace() for chr in text)
                sentences_len = sum([sum(not chr.isspace() for chr in s) for s in sentences])
                self.assertEqual(sentences_len, text_len)

                # check if all sentences have correct beginning
                for i, s in enumerate(sentences):
                    expected_beginning = expected[i]
                    self.assertTrue(s.startswith(expected_beginning), msg=f'{i}: "{s}" have incorrect beginning')

    def test_list_tokenization(self):
        sc = SectionsComparator()
        file_path = os.path.join(self.resources_path, 'list.txt')
        with open(file_path, 'r', encoding='UTF-8') as src:
            text = src.read()

        sentences = sc._tokenize_text(text)
        self.assertEqual(len(sentences), 30)
        for i, s in enumerate(sentences):
            if i % 2 == 1:
                self.assertTrue(s, '\n')
            else:
                self.assertTrue(s.startswith('*'))

    def test_split_on_quotes(self):
        test_cases = [
            ('{{quote|bb}}', ['{{quote|bb}}']),
            ('aa {{quote|bb}}', ['aa', '{{quote|bb}}']),
            ('aa {{quote|bb}} cc', ['aa', '{{quote|bb}}', 'cc']),
            ('broken {{quote] cc', ['broken {{quote] cc']),
            ('broken2 {{quote cc}} d', ['broken2 {{quote cc}} d']),
            ('nested {{quote|bb {{quote|cc}}}} dd', ['nested', '{{quote|bb {{quote|cc}}}}', 'dd']),
            ('spaaaces {{quote|bb {{quote|cc}}}}    dd', ['spaaaces', '{{quote|bb {{quote|cc}}}}', 'dd']),
            ('following text {{quote|bb {{quote|cc}}}}dd', ['following text', '{{quote|bb {{quote|cc}}}}', 'dd']),
        ]

        sc = SectionsComparator()
        for i, tc in enumerate(test_cases):
            with self.subTest(f'Test {i}'):
                expected_sentences = tc[1]
                text = tc[0]

                sentences = sc._tokenize_text(text)

                self.assertEqual(len(sentences), len(expected_sentences))
                for j in range(len(sentences)):
                    self.assertEqual(sentences[j], expected_sentences[j])

    def test_problematic_patterns(self):
        test_cases = [
            'problematic_pattern_smily.txt',
            'problematic_pattern_!.txt'
        ]

        parser = WikiMarkupParser()
        comparator = SectionsComparator()

        for tc in test_cases:
            with open(os.path.join(self.resources_path, tc), 'r', encoding='UTF-8') as src:
                text = src.read()
            sections = parser.parse_wiki_markup(text)

            with self.assertRaises(ProblematicTokenizationPatternException):
                comparator.tokenize_sections(sections)


if __name__ == '__main__':
    unittest.main()
