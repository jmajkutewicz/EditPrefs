import unittest
from pathlib import Path

from wikiprefs.markup.markup_processing import WikiBrokenMarkupException, WikiMarkupParser


class TestDemonstrateSubtest(unittest.TestCase):
    resources_base = Path(__file__).parent / 'resources'
    parser = WikiMarkupParser()

    def test_broken_sections_tree(self):
        with open(self.resources_base / 'markup/sections.txt', 'r', encoding='UTF-8') as src:
            text = src.read()

        sections = self.parser.parse_wiki_markup(text)
        self.assertEqual(len(sections), 12)

        broken_section = sections[9]
        self.assertEqual(broken_section[0], 'forth section (broken) =')
        self.assertEqual(broken_section[1], 'd')

        broken_section_subsection = sections[10]
        self.assertEqual(broken_section_subsection[0], 'forth section (broken) = // subsection 4.1')
        self.assertEqual(broken_section_subsection[1], 'dd')

    def test_unclosed_tag(self):
        test_cases = [
            'unclosed_tag',
            'unclosed_template',
            'unclosed_tag_in_link'
        ]
        for tc in test_cases:
            with self.subTest(tc):
                with open(self.resources_base / f'markup/{tc}.txt', 'r', encoding='UTF-8') as src:
                    text = src.read()

                with self.assertRaises(WikiBrokenMarkupException):
                    self.parser.parse_wiki_markup(text)

    def test_markup_processing(self):
        test_cases = [
            ('1 Line articles', '1_line'),
            ('Acupuncture article', 'acupunture'),
            ('And Justice for All intro', 'And Justice for All'),
            ('Article starting with template', 'sonic'),
            ('Template inside tag', 'Banded broadbill'),
            ('Pre-formated text', 'Banksia ericifolia'),
            ('Battle_of_Sluys articles', 'Battle_of_Sluys'),
            ('Comments removal', 'comments'),
            ('Dish-bearers and butlers in Anglo-Saxon England article',
             'Dish-bearers and butlers in Anglo-Saxon England'),
            ('File links', 'file'),
            ('Anchor in section heading', 'Gabriel Pleydell'),
            ('Comment in section heading', 'Germany'),
            ('Headline with tag', 'headline'),
            ('List', 'list'),
            ('Math', 'math'),
            ('Nat fs (multi line template)', 'nat_fs'),
            ('File link', 'No. 1 Flying Training School RAAF'),
            ('Ref tag', 'ref'),
            ('References', 'references'),
            ('Simple templates', 'simple_templates'),
            ('Multiple templates', 'templates'),
            ('Unclosed image link', 'unclosed_link'),
            ('US presidential ticket box (multi line template)', 'us_presidential_ticket_box'),
            # FIXME: handle all templates inside tags correctly:
            # ('Table inside div tag', 'table_inside_div'),
        ]

        for tc in test_cases:
            filename = tc[1]
            with self.subTest(tc[0], filename=filename):
                with open(self.resources_base / f'markup/src/{filename}.txt', 'r', encoding='UTF-8') as f:
                    markup = f.read()
                with open(self.resources_base / f'markup/cleaned/{filename}.txt', 'r', encoding='UTF-8') as f:
                    expected = f.read()

                sections = self.parser.parse_wiki_markup(markup)
                cleaned_text = []
                for section in sections:
                    cleaned_text.append(f'==={section[0]}===\n')
                    cleaned_text.append(section[1])
                    cleaned_text.append('\n')
                self.assertEqual(''.join(cleaned_text), expected)


if __name__ == '__main__':
    unittest.main()
