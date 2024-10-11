import os.path
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from wikiprefs.find_npov_edits import process_metahistory_stub, PageHistoryConsumer, process_metahistory_stubs_xml


class FindNPOVEditsTest(unittest.TestCase):
    RESOURCES_DIR = Path(__file__).parent / 'resources'

    def test_revisions_count(self):
        pages2revs = {
            'AccessibleComputing': 10,
            'Anarchism': 12894,
        }

        class TestPageHistoryConsumer(PageHistoryConsumer):
            def on_page_processed(inner_self) -> None:
                revisions = inner_self.revision_chain_creator.on_page_processed()

                page = inner_self.curr_page.title
                self.assertTrue(page in pages2revs)

                expected_revs_count = pages2revs[page]
                self.assertEqual(expected_revs_count, len(revisions))

        metahistory_stub_name = 'meta-history-stub'
        metahistory_stub_filepath = self.RESOURCES_DIR / 'npov_edits' / f'{metahistory_stub_name}.xml.gz'
        temp_dir = tempfile.mkdtemp()

        consumer = TestPageHistoryConsumer(os.path.join(temp_dir, 'test.csv'))

        process_metahistory_stubs_xml(consumer, metahistory_stub_filepath)

    def test_save_npov_edits(self):
        metahistory_stub_name = 'meta-history-stub'
        metahistory_stub_filepath = self.RESOURCES_DIR / 'npov_edits' / f'{metahistory_stub_name}.xml.gz'
        temp_dir = tempfile.mkdtemp()
        csv_file_path = os.path.join(temp_dir, 'test.csv')

        process_metahistory_stub(metahistory_stub_name, metahistory_stub_filepath, csv_file_path)

        df = pd.read_csv(csv_file_path)
        self.assertEqual(df[df['page_title'] == 'AccessibleComputing'].shape[0], 0)
        self.assertEqual(df[df['page_title'] == 'Anarchism'].shape[0], 207)

    def test_page_re(self):
        csv_file_path = os.path.join(tempfile.mkdtemp(), 'test.csv')
        consumer = PageHistoryConsumer(csv_file_path)
        test_cases = [
            'Help:About help pages',
            'Help:Maintenance template removal',
            'Template:Dispute templates',
        ]

        for tc in test_cases:
            with self.subTest(f'Testing: {tc}'):
                m = consumer.title_re.match(tc)
                self.assertIsNotNone(m)


if __name__ == '__main__':
    unittest.main()
