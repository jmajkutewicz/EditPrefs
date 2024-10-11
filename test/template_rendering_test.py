import unittest
from pathlib import Path

import pandas as pd
import yaml

from wikiprefs.collect_retained_diffs import DiffRow
from wikiprefs.create_dataset import render_templates
from wikiprefs.markup.caching_template_renderer import CachingTemplateRenderer


class TemplateRenderingTest(unittest.TestCase):
    RESOURCES_DIR = Path(__file__).parent / 'resources'
    new_text_format = 'new text start {} new text end'

    def test_invalid_templates_rendering(self):
        test_cases = [
            ('{{Inflation-year|US-NGDPPC}}', None),
            ('{{Harvnp|Sfn|Ankel-Simons|2007|p=544}}', None),
            ('{{Nihongo|\u500D\u97F3}}', None),
            ('{{Asia 576 CE|right|The Chalukyas and contemporary polities {{c.|576}}}}', None),
        ]
        sound = (r'{{Block indent|<score sound="1"> { '
                 r'\new PianoStaff \with { instrumentName = "Pno." } << '
                 r'\new Staff \relative c,, { \clef bass \key aes \major \time 2/4 \tempo '
                 '"Fast" 4 = 152 f16\\p( c\' aes c bes-. r8)\\bar "|" } '
                 r'>> } </score>}}')
        test_cases.append((sound, None))

        df = self._render_templates(test_cases)

        self.assertEqual(df.shape[0], 0)

    def test_template_rendering(self):
        test_cases = [
            ('{{convert|1000|km2}}', '1,000 square kilometres (390 sq mi)'),
            ('{{As of|2012|lc=on}}', 'as of 2012'),
            ('{{Bibleref2|Luke|7:36\u201350}}', 'Luke 7:36–50'),
            ('{{cbb link|2007|sex=men|team=Creighton Bluejays|school=Creighton University|title=Creighton}}',
             'Creighton'),
            ('{{chem2|Rh+ + [(\\h{5}C5H5)2M] \u2192 M + [(\\h{5}C5H5)2Rh]+}}', 'Rh + [(η-C5H5)2M] → M + [(η-C5H5)2Rh]'),
            ('{{circa|1375}}', 'c. 1375'),
            ('{{convert|24|*|12|*|20|in|cm|adj=on}}', '24×12×20-inch (61×30×51 cm)'),
            ('{{frac|1|3|4}}', '1+3⁄4'),
            ('{{HMS|Iveston|M1151|6}}', 'HMS Iveston'),
            ('{{Inflation|US|0.075|1896|r=1}}', '2.7'),
            ('{{IPAc-en|\u02c8|b|\u025b|k|\u0259|r|z}}', '/ˈbɛkərz/'),
            ('{{lang-ar|\u0648\u0642\u0639\u0629 \u0627\u0644\u0634\u0639\u0628|Waq\u02bfat al-Sh\u02bfib}}',
             'Arabic: وقعة الشعب, romanized: Waqʿat al-Shʿib'),
            ('{{lang|zh-Hant-TW|\u5b97\u6fa4}}', '宗澤'),
            ('{{nowrap|5,000,000 listeners}}', '5,000,000 listeners'),
            ('{{sclass2|250t|torpedo boat|0}}', '250t-class'),
            ('{{To USD round|14000000|MUS|year=2018}}', '410,000'),
            ('{{transliteration|ar|[[mawali]]}}', 'mawali'),
            ('{{\u00a3sd|s=7|d=6}}', '0.38'),
            ('{{Ussc|578|___|2016|el=no}}', '578 U.S. ___ (2016)'),
            ('{{ ill | Glenn M. Shea | WD = Q22105609 }}', 'Glenn M. Shea  [Wikidata]'),
            ('{{Coord|37.7757|-122.451|display=inline}}',
             '37°46′33″N 122°27′04″W﻿ / ﻿37.7757°N 122.451°W﻿ / 37.7757; -122.451'),
            ('{{Frac|10|3|4}}', '10+3⁄4'),
            ('{{GBP|170&nbsp;million|link=yes}}', '£170 million'),
            ('{{HMAS|Sydney|1912|6}}', 'HMAS Sydney'),
            ('{{Lang|de|[[Christus, der ist mein Leben, BWV 95|''''Christus, der ist mein Leben'''', BWV 95]]}}',
             'Christus, der ist mein Leben, BWV 95'),
        ]

        # test_render_template()

        df = self._render_templates(test_cases)

        for i, tc in enumerate(test_cases):
            expected_template = tc[1]
            expected_text = self.new_text_format.format(expected_template)
            rendered_text = df.iloc[i]['new_text']
            self.assertEqual(rendered_text, expected_text)

    def test_escape_html(self):
        template2expected = {
            '{{As of|2012|lc=on}}': 'as of 2012',
            '{{Bibleref2|Luke|7:36–50}}': 'Luke 7:36–50',
            '{{HMS|Iveston|M1151|6}}': 'HMS Iveston',
            '{{IPA-id|ˈsɔrɡa kə ˈtudʒu|}}': '[ˈsɔrɡa kə ˈtudʒu]',
            '{{IPAc-en|ˈ|b|ɛ|k|ə|r|z}}': '/ˈbɛkərz/',
            '{{Inflation|US|0.075|1896|r=1}}': '2.7',
            '{{To USD round|14000000|MUS|year=2018}}': '410,000',
            '{{Ussc|578|___|2016|el=no}}': '578 U.S. ___ (2016)',
            '{{cbb link|2007|sex=men|team=Creighton Bluejays|school=Creighton University|title=Creighton}}': 'Creighton',
            '{{chem2|Rh+ + [(\h{5}C5H5)2M] → M + [(\h{5}C5H5)2Rh]+}}': 'Rh + [(η-C5H5)2M] → M + [(η-C5H5)2Rh]',
            '{{circa|1375}}': 'c. 1375',
            '{{convert|0.3|m|ft|sigfig=1|abbr=on}}': '0.3 m (1 ft)',
            '{{convert|0.5|oz}}': '0.5 ounces (14 g)',
            '{{convert|1000|km2}}': '1,000 square kilometres (390 sq mi)',
            '{{convert|13|x|5|m|ft|abbr=on}}': '13 m × 5 m (43 ft × 16 ft)',
            '{{convert|18|x|7|m|ft|abbr=on}}': '18 m × 7 m (59 ft × 23 ft)',
            '{{convert|24|*|12|*|20|in|cm|adj=on}}': '24×12×20-inch (61×30×51 cm)',
            '{{convert|2|m|ft|sigfig=1|abbr=on}}': '2 m (7 ft)',
            '{{convert|6|m|ft|abbr=on}}': '6 m (20 ft)',
            '{{convert|7700|ha|acre|sigfig=2|adj=on}}': '7,700-hectare (19,000-acre)',
            '{{frac|1|3|4}}': '1+3⁄4',
            '{{lang-ar|وقعة الشعب|Waqʿat al-Shʿib}}': 'Arabic: وقعة الشعب, romanized: Waqʿat al-Shʿib',
            '{{lang|lat|caesar}}': 'caesarcode: lat promoted to code: la ',
            '{{lang|zh-Hant-TW|宗澤}}': '宗澤',
            '{{nowrap|5,000,000 listeners}}': '5,000,000 listeners',
            '{{sclass2|250t|torpedo boat|0}}': '250t-class',
            '{{transliteration|ar|[[mawali]]}}': 'mawali',
            '{{£sd|s=7|d=6}}': '0.38',
            '{{ ill | Glenn M. Shea | WD = Q22105609 }}': 'Glenn M. Shea  [Wikidata]',
            '{{As of|2011|07}}': 'As of July 2011',
            '{{Asia 576 CE|right|The Chalukyas and contemporary polities {{c.|576}}}}': None,
            '{{BoM TC Database}}': '',
            '{{Coord|37.7757|-122.451|display=inline}}': '37°46′33″N 122°27′04″W﻿ / ﻿37.7757°N 122.451°W﻿ / 37.7757; -122.451',
            '{{Frac|10|3|4}}': '10+3⁄4',
            '{{GBP|170&nbsp;million|link=yes}}': '£170 million',
            '{{HMAS|Sydney|1912|6}}': 'HMAS Sydney',
            '{{Harvnp|Sfn|Ankel-Simons|2007|p=544}}': None,
            '{{Lang|de|[[Christus, der ist mein Leben, BWV 95|Christus, der ist mein Leben, BWV 95]]}}': 'Christus, der ist mein Leben, BWV 95',
            '{{Nihongo|倍音}}': None,
            '{{Sfn|Brown|1948|pp=300–301}}': '',
            '{{efn|1 Corinthians 15:41}}': '',
        }
        sound = ('{{Block indent|<score sound="1"> \\relative c\'\' { \\set Staff.midiInstrument = #"bassoon" '
                 '\\clef treble \\numericTimeSignature \\time 4/4 \\tempo "Lento" 4 = 50 \\stemDown c4\\fermata(_"solo ad lib." '
                 '\\grace { b16[( c] } b g e b\' \\times 2/3 { a8)\\fermata } } </score>}}')

        template2expected[sound] = None

        with open(self.RESOURCES_DIR / 'rendered_templates_with_html.yml') as f:
            y = yaml.load(f, Loader=yaml.SafeLoader)
            template2html = y['templates']
        print(template2html.keys())

        renderer = CachingTemplateRenderer()

        for k in template2expected.keys():
            expected = template2expected[k]
            decoded = renderer._decode_html(template2html[k])
            self.assertEqual(decoded, expected)

    def _render_templates(self, test_cases):
        diffs = []
        for i, tc in enumerate(test_cases):
            new_text = self.new_text_format.format(tc[0])
            diffs.append(self._create_diff_row(i, 'old', new_text))

        df = pd.DataFrame([p.__dict__ for p in diffs])
        render_templates(df)

        return df

    def _create_diff_row(self, i, old_text, new_text):
        row = DiffRow(
            page_title=f'test_{i}',
            page_id=i,
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


if __name__ == '__main__':
    unittest.main()
