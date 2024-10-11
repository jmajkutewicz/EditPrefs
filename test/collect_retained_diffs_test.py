import os
import unittest
from pathlib import Path

import pandas as pd

from wikiprefs.collect_retained_diffs import collect_retained_diffs, process_article, setup_arg_parser


class CollectRetainedDiffsTest(unittest.TestCase):
    RESOURCES_DIR = Path(__file__).parent / 'resources/diffs_collection'
    RESULTS_DIR = Path(__file__).parent / 'resources/diffs_collection/results'

    def test_collect_retained_diffs(self):
        test_files = {
            'Dish-bearers and butlers in Anglo-Saxon England': 21,
            'Science Fiction Adventures (1956 magazine)': 19,
            'Seorsumuscardinus': 14,
            'test_xml': 4,
        }
        self._clear_results_directory()

        parser = setup_arg_parser()
        config = parser.parse_args(['--src', str(self.RESOURCES_DIR), '--dst', str(self.RESULTS_DIR)])

        collect_retained_diffs(config)

        for tc in test_files.keys():
            with self.subTest(f'Checking "{tc}" diffs'):
                expected_diffs_count = test_files[tc]

                csv_path = os.path.join(self.RESULTS_DIR, f'{tc}.csv')
                df = pd.read_csv(csv_path)

                self.assertEqual(expected_diffs_count, df.shape[0])

    def test_process_article(self):
        # revision_id: (old_text, new_text)
        test_cases = {
            2: ('AA bb cc', 'aa bb cc'),
            # 3: ('qq ww ee', 'qq ww EE'),
            4: ('qq ww EE\naa SS dd', 'qq ww EE\naa ss dd'),
            5: ('aa bb cc\nii JJ kk', 'aa bb cc\nii jj kk'),
            6: ('qq ww EE', 'qq ww ee'),
        }

        dst_csv_path = self._collect_diffs('test_xml')

        df = pd.read_csv(dst_csv_path)
        self.assertEqual(df.shape[0], 4)

        for tc in test_cases.keys():
            with self.subTest(f'Testing revision {tc}'):
                expected_old_text, expected_new_text = test_cases[tc]

                revision_row = df[df['rev_id'] == tc]
                self.assertIsNotNone(revision_row)

                new_text = revision_row['new_text'].iloc[0]
                old_text = revision_row['old_text'].iloc[0]

                self.assertEqual(new_text, expected_new_text)
                self.assertEqual(old_text, expected_old_text)

    def test_discography_list_diff(self):
        dst_csv_path = self._collect_diffs('Alice in Chains')

        df = pd.read_csv(dst_csv_path)
        self.assertEqual(df.shape[0], 1)

    def test_cast_list_diff(self):
        dst_csv_path = self._collect_diffs('Aliens__cast')

        df = pd.read_csv(dst_csv_path)
        self.assertEqual(df.shape[0], 1)

    def test_multiple_lists_diff(self):
        dst_csv_path = self._collect_diffs('Baseball_multiple_lists')

        df = pd.read_csv(dst_csv_path)
        self.assertEqual(df.shape[0], 1)

    def test_article_start_with_edited_link(self):
        dst_csv_path = self._collect_diffs('1988 World Snooker Championship')

        df = pd.read_csv(dst_csv_path)
        df = df[df['rev_id'] == 2]
        self.assertEqual(df.shape[0], 1)

        for i, row in df.iterrows():
            old_text = row['old_text']
            self.assertEqual(old_text,
                             'The defending champion was Steve Davis, who had previously won the World '
                             'Championship four times.He met the 1979 champion Terry Griffiths in the final, '
                             'which was a best-of-35-frames match.')

            new_text = row['new_text']
            self.assertEqual(new_text,
                             'The defending champion was Steve Davis, who had previously won the World '
                             'Championship four times.He met the 1979 champion Terry Griffiths in the final, '
                             'which was a best-of-35-{{cuegloss|frames}} match.')

    def test_math_paragraph(self):
        dst_csv_path = self._collect_diffs('Leonhard Euler')

        df = pd.read_csv(dst_csv_path)
        df = df[df['rev_id'] == 2]
        df = df[df['section'] == 'Contributions to mathematics and physics // Analysis']
        for i, row in df.iterrows():
            self.assertTrue('<math>' in row['new_text'])
            self.assertTrue('<math>' in row['old_text'])

    def test_math_paragraph_2(self):
        dst_csv_path = self._collect_diffs('Logarithm')

        df = pd.read_csv(dst_csv_path)
        df = df[df['rev_id'] == 2]
        self.assertEqual(df.shape[0], 56)

        df = df[df['section'] == 'Generalizations // Complex logarithm']
        # FIXME: detect broken markup in a single seciton

    def test_reverts(self):
        dst_csv_path = self._collect_diffs('test_reverts')

        df = pd.read_csv(dst_csv_path)
        self.assertEqual(df.shape[0], 1)

        row = df.iloc[0]
        self.assertEqual(row['rev_id'], 6)
        self.assertEqual(row['prev_rev_id'], 1)
        self.assertEqual(row['old_text'], 'aa bb cc\nxxyy zz')
        self.assertEqual(row['new_text'], 'aa bb cc\nxxyy zz ww')

    def test_text_only_removed(self):
        dst_csv_path = self._collect_diffs('Armenian genocide__removed_text')

        df = pd.read_csv(dst_csv_path)
        df = df[df['rev_id'] == 2]
        self.assertEqual(df.shape[0], 0)

    def test_text_only_removed_and_broken(self):
        dst_csv_path = self._collect_diffs('Augustus')

        df = pd.read_csv(dst_csv_path)
        df = df[df['rev_id'] == 2]
        self.assertEqual(df.shape[0], 0)

    def test_broken_latest_revision(self):
        dst_csv_path = self._collect_diffs('Germany')

        df = pd.read_csv(dst_csv_path)
        self.assertEqual(df.shape[0], 2)

    def test_broken_all_revision(self):
        dst_csv_path = self._collect_diffs('Elvis Presley')

        df = pd.read_csv(dst_csv_path)
        self.assertEqual(df.shape[0], 0)

    def test_diff_context(self):
        test_cases = [
            # https://en.wikipedia.org/w/index.php?diff=prev&oldid=1008293045
            ('Aliens__context', 1,
             [
                 (
                     # old text:
                     'On an approximately $18.5 million budget, "Aliens" began principal photography in September 1985.'
                     'Like its development, filming was tumultuous and rife with conflicts between Cameron and the '
                     'British crew at Pinewood Studios over their work habits and Cameron\'s relative inexperience.',
                     # new text:
                     'On an approximately $18.5 million budget, "Aliens" began principal photography in September 1985.'
                     'Like its development, filming was tumultuous and rife with conflicts between Cameron and the '
                     'British crew at Pinewood Studios.'
                 )
             ]),
            # https://en.wikipedia.org/w/index.php?diff=prev&oldid=876754812
            ('Aaliyah__context', 1, [
                (
                    # old text:
                    'She has been credited for helping redefine contemporary R&B, pop and hip hop, earning her the '
                    'nicknames "Princess of R&B" and "Queen of Urban Pop".She is listed by "Billboard" as the tenth '
                    'most successful female R&B artist of the past 25 years, and the 27th most successful '
                    'R&B artist in history.',
                    # new text:
                    'She has been credited for helping redefine contemporary R&B, pop and hip hop, earning her the '
                    'nicknames "Princess of R&B" and "Queen of Urban Pop"."Billboard" lists her as the tenth '
                    'most successful female R&B artist of the past 25 years, and the 27th most successful in history.'
                ),
            ]),
            # https://en.wikipedia.org/w/index.php?diff=prev&oldid=1117784527
            ('Alfred Russel Wallace__context', 1, [
                (
                    # old text:
                    'From 1854 to 1862, Wallace travelled around the islands of the Malay Archipelago or East Indies '
                    '(now Singapore, Malaysia and Indonesia).His main objective "was to obtain specimens of natural '
                    'history, both for my private collection and to supply duplicates to museums and amateurs".'
                    'In addition to Allen, he "generally employed one or two, and sometimes three Malay servants" '
                    'as assistants, and paid large numbers of local people to bring specimens.His total was 125,660 '
                    'specimens, most of which were insects including more than 83,000 beetles, Several thousand of the '
                    'specimens represented species new to science, including the gliding tree frog "Rhacophorus '
                    'nigropalmatus", known as Wallace\'s flying frog, provided by a Chinese workman at a mining site, '
                    'who described its behaviour to Wallace.Overall, more than thirty men worked for him at some stage '
                    'as full-time paid collectors.He also hired guides, porters, cooks and boat crews, so was '
                    'assisted by an estimated 1,200 individuals.',
                    # new text:
                    'From 1854 to 1862, Wallace travelled around the islands of the Malay Archipelago or East Indies '
                    '(now Singapore, Malaysia and Indonesia).His main objective "was to obtain specimens of natural '
                    'history, both for my private collection and to supply duplicates to museums and amateurs".'
                    'In addition to Allen, he "generally employed one or two, and sometimes three Malay servants" as '
                    'assistants, and paid large numbers of local people at various places to bring specimens.'
                    'His total was 125,660 specimens, most of which were insects including more than 83,000 beetles, '
                    'Several thousand of the specimens represented species new to science, including the gliding tree '
                    'frog "Rhacophorus nigropalmatus", known as Wallace\'s flying frog, provided by a Chinese workman '
                    'at a mining site, who described its behaviour to Wallace.Overall, more than thirty men worked for '
                    'him at some stage as full-time paid collectors.He also hired guides, porters, cooks and boat '
                    'crews, so well over 100 individuals worked for him.'
                )
            ]),
            # https://en.wikipedia.org/w/index.php?diff=prev&oldid=621232997
            ('Alice in Chains__quote', 1, [
                (
                    # old text:
                    'Although Alice in Chains has been labeled grunge by the mainstream media, '
                    'Jerry Cantrell identifies the band as primarily [Rock music].'
                    'He told "Guitar World" in 1996, "We\'re a lot of different things ...I don\'t quite know what '
                    'the mixture is, but there\'s definitely metal, blues, rock and roll, maybe a touch of punk.'
                    'The metal part will never leave, and I never want it to".The "Edmonton Journal" has stated, '
                    '"Living and playing in Seattle might have got them the grunge tag, '
                    'but they\'ve always pretty much been a classic rock band to the core." '
                    'Over the course of their career, the band\'s sound has also been described as alternative rock, '
                    'sludge metal, doom metal, dark metal, drone rock, hard rock, and alternative rock.'
                    'Regarding the band\'s constant categorization by the media, Cantrell stated '
                    '"When we first came out we were metal.Then we started being called alternative metal.'
                    'Then grunge came outand then we were hard rock.And now, since we\'ve started doing this again '
                    'I\'ve seen us listed as: hard rock, alternative, alternative rock.',
                    # new text:
                    'Although Alice in Chains has been labeled grunge by the mainstream media, '
                    'Jerry Cantrell identifies the band as primarily heavy metal.'
                    'He told "Guitar World" in 1996, "We\'re a lot of different things ...I don\'t quite know what '
                    'the mixture is, but there\'s definitely metal, blues, rock and roll, maybe a touch of punk.'
                    'The metal part will never leave, and I never want it to".The "Edmonton Journal" has stated, '
                    '"Living and playing in Seattle might have got them the grunge tag, '
                    'but they\'ve always pretty much been a classic metal band to the core." '
                    'Over the course of their career, the band\'s sound has also been described as alternative metal, '
                    'sludge metal, doom metal, dark metal, drone rock, hard rock, and alternative rock.'
                    'Regarding the band\'s constant categorization by the media, Cantrell stated '
                    '"When we first came out we were metal.Then we started being called alternative metal.'
                    'Then grunge came outand then we were hard rock.And now, since we\'ve started doing this again '
                    'I\'ve seen us listed as: hard rock, alternative, alternative metal and just straight metal.'
                )
            ]),
        ]

        for tc in test_cases:
            with self.subTest(f'Testing {tc[0]}'):
                filename = tc[0]
                expected_changes = tc[1]
                expected_texts = tc[2]

                dst_csv_path = self._collect_diffs(filename)
                df = pd.read_csv(dst_csv_path)
                df = df[df['rev_id'] == 2]
                self.assertEqual(df.shape[0], expected_changes)

                for i, row in df.iterrows():
                    expected_text = expected_texts[i]
                    old_text = row['old_text']
                    new_text = row['new_text']
                    self.assertEqual(old_text, expected_text[0])
                    self.assertEqual(new_text, expected_text[1])

    def _collect_diffs(self, metahistory_file_name: str):
        self._clear_results_directory()

        src_xml_path = os.path.join(self.RESOURCES_DIR, f'{metahistory_file_name}.xml')
        dst_csv_path = os.path.join(self.RESULTS_DIR, f'{metahistory_file_name}.csv')
        process_article(src_xml_path, dst_csv_path, 3)
        return dst_csv_path

    def _clear_results_directory(self):
        for filename in os.listdir(self.RESULTS_DIR):
            file_path = os.path.join(self.RESULTS_DIR, filename)
            if filename != '.gitkeep' and os.path.isfile(file_path):
                os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
