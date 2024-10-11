import os.path
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from wikiprefs.collect_npov_diffs import PageHistoryConsumer
from wikiprefs.find_npov_edits import NpovEdit
from wikiprefs.utils.xml_utils import Page, Revision


class TestPageHistoryConsumer(unittest.TestCase):

    def setUp(self):
        self.pages_to_extract = {"1"}
        self.page2edits = {
            "1": [NpovEdit(page_id='1', page_title='page1', rev_id='1', parent_rev_id='0', comment='test')]}
        temp_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(temp_dir, "test.csv")

    @patch("csv.DictWriter")
    def test_initialization_creates_csv(self, mock_dict_writer):
        PageHistoryConsumer(self.pages_to_extract, self.page2edits, self.csv_file)

        self.assertTrue(os.path.exists(self.csv_file))
        mock_dict_writer.assert_called_once()

    def test_revision_deletes_article(self):
        consumer = PageHistoryConsumer(self.pages_to_extract, self.page2edits, self.csv_file)
        parent_rev = MagicMock(text_size=1000)
        rev = MagicMock(text_size=50)

        self.assertTrue(consumer._revision_deletes_article(rev, parent_rev))

    def test_on_page_started(self):
        consumer = PageHistoryConsumer(self.pages_to_extract, self.page2edits, self.csv_file)

        consumer.on_page_started()

        self.assertIsNone(consumer.current_page)
        self.assertTrue(consumer.skip_page)
        self.assertIsNone(consumer.current_edits)
        self.assertIsNone(consumer.current_edit)
        self.assertEqual(consumer.current_i, -1)
        self.assertIsNone(consumer.current_parent_rev)

    def test_on_revision_processed_skip_page(self):
        page_to_skip = MagicMock(id="3")
        revision = MagicMock(id="1")
        consumer = PageHistoryConsumer(self.pages_to_extract, self.page2edits, self.csv_file)

        consumer.on_page_started()
        consumer.on_revision_processed(page_to_skip, revision)

        self.assertTrue(consumer.skip_page)

        consumer.on_revision_processed(MagicMock(id="1"), MagicMock(id="0"))
        self.assertIsNone(consumer.current_parent_rev)

    @patch.object(PageHistoryConsumer, "_save_diffs")
    def test_on_revision_processed_no_parent(self, mock_save_diffs):
        consumer = PageHistoryConsumer(self.pages_to_extract, self.page2edits, self.csv_file)
        consumer.on_page_started()
        page = MagicMock(id="1", title="page1")
        revision = MagicMock(id="1")

        consumer.on_revision_processed(page, revision)

        mock_save_diffs.assert_not_called()
        self.assertIsNone(consumer.current_edit)
        self.assertIsNone(consumer.current_parent_rev)
        self.assertEqual(consumer.current_i, 1)
        self.assertTrue(consumer.skip_page)

    @patch.object(PageHistoryConsumer, "_save_diffs")
    def test_save_diffs_normal_revision(self, mock_save_diffs):
        consumer = PageHistoryConsumer(self.pages_to_extract, self.page2edits, self.csv_file)

        page = MagicMock(id="1", title="page1")
        parent_revision = MagicMock(id="0")
        revision = MagicMock(id="1")

        consumer.on_revision_processed(page, parent_revision)
        consumer.on_revision_processed(page, revision)

        mock_save_diffs.assert_called_once_with(page, parent_revision, revision)
        self.assertIsNone(consumer.current_edit)
        self.assertIsNone(consumer.current_parent_rev)
        self.assertEqual(consumer.current_i, 1)
        self.assertTrue(consumer.skip_page)

    @patch.object(PageHistoryConsumer, "_save_diffs")
    def test_ignored_revision(self, mock_save_diffs):
        consumer = PageHistoryConsumer(self.pages_to_extract, self.page2edits, self.csv_file)

        page = MagicMock(id="1", title="page1")
        parent_revision = MagicMock(id="0")
        revision = MagicMock(id="1")

        consumer.on_revision_processed(page, MagicMock(id="-1"))
        self.assertIsNone(consumer.current_parent_rev)

        consumer.on_revision_processed(page, parent_revision)
        self.assertIsNotNone(consumer.current_parent_rev)

        consumer.on_revision_processed(page, MagicMock(id="3"))
        self.assertIsNotNone(consumer.current_parent_rev)
        mock_save_diffs.assert_not_called()

        consumer.on_revision_processed(page, revision)
        mock_save_diffs.assert_called_once_with(page, parent_revision, revision)

    @patch.object(PageHistoryConsumer, "_save_diffs")
    def test_multiple_edits(self, mock_save_diffs):
        page = MagicMock(id="1", title="page1")
        page2edits = {
            "1": [
                NpovEdit(page_id='1', page_title='page1', rev_id='1', parent_rev_id='0', comment='test'),
                NpovEdit(page_id='1', page_title='page1', rev_id='5', parent_rev_id='4', comment='test 2')
            ]
        }
        consumer = PageHistoryConsumer(self.pages_to_extract, page2edits, self.csv_file)

        consumer.on_revision_processed(page, MagicMock(id="0"))
        consumer.on_revision_processed(page, MagicMock(id="1"))
        mock_save_diffs.assert_called_once()
        mock_save_diffs.reset_mock()
        self.assertEqual(consumer.current_edit, page2edits['1'][1])

        consumer.on_revision_processed(page, MagicMock(id="2"))
        consumer.on_revision_processed(page, MagicMock(id="3"))
        self.assertIsNone(consumer.current_parent_rev)
        mock_save_diffs.assert_not_called()

        consumer.on_revision_processed(page, MagicMock(id="4"))
        consumer.on_revision_processed(page, MagicMock(id="5"))
        mock_save_diffs.assert_called_once()
        self.assertTrue(consumer.skip_page)

        consumer.on_revision_processed(page, MagicMock(id="6"))

    @patch.object(PageHistoryConsumer, "_save_diffs")
    def test_revision_is_both_child_and_parent(self, mock_save_diffs):
        page = MagicMock(id="1", title="page1")
        page2edits = {
            "1": [
                NpovEdit(page_id='1', page_title='page1', rev_id='1', parent_rev_id='0', comment='test'),
                NpovEdit(page_id='1', page_title='page1', rev_id='2', parent_rev_id='1', comment='test 2')
            ]
        }
        consumer = PageHistoryConsumer(self.pages_to_extract, page2edits, self.csv_file)

        first_rev = MagicMock(id="0")
        second_rev = MagicMock(id="1")
        third_rev = MagicMock(id="2")

        consumer.on_revision_processed(page, first_rev)
        consumer.on_revision_processed(page, second_rev)
        mock_save_diffs.assert_called_once()
        mock_save_diffs.reset_mock()
        self.assertEqual(consumer.current_edit, page2edits['1'][1])
        self.assertEqual(consumer.current_parent_rev, second_rev)

        consumer.on_revision_processed(page, third_rev)
        mock_save_diffs.assert_called_once()
        self.assertTrue(consumer.skip_page)

    @patch.object(PageHistoryConsumer, "_save_diffs")
    def test_multiple_pages(self, mock_save_diffs):
        page2edits = {
            "1": [
                NpovEdit(page_id='1', page_title='page1', rev_id='1', parent_rev_id='0', comment='test'),
            ],
            "2": [
                NpovEdit(page_id='2', page_title='page2', rev_id='11', parent_rev_id='10', comment='test2'),
            ]
        }
        consumer = PageHistoryConsumer(self.pages_to_extract, page2edits, self.csv_file)

        page = MagicMock(id="1", title="page1")
        consumer.on_page_started()
        consumer.on_revision_processed(page, MagicMock(id="0"))
        consumer.on_revision_processed(page, MagicMock(id="1"))
        mock_save_diffs.assert_called_once()
        mock_save_diffs.reset_mock()

        page = MagicMock(id="2", title="page2")
        consumer.on_page_started()
        consumer.on_revision_processed(page, MagicMock(id="10"))
        consumer.on_revision_processed(page, MagicMock(id="11"))
        mock_save_diffs.assert_called_once()
        mock_save_diffs.reset_mock()

    def test_create_diffs(self):
        consumer = PageHistoryConsumer(self.pages_to_extract, self.page2edits, self.csv_file)

        page = Page('page 1', '1')
        v1_text, v2_text = self.load_text('modified_long_suffix')
        rev_1 = Revision(
            id='0',
            parent_id='',
            timestamp='2002-12-11T09:39:56Z',
            is_minor=False,
            contributor='user',
            comment='comment',
            sha1='sha1version1',
            text_size=123,
            text=v1_text
        )
        rev_2 = Revision(
            id='1',
            parent_id='0',
            timestamp='2004-12-11T09:39:56Z',
            is_minor=False,
            contributor='user',
            comment='update comment',
            sha1='sha1version2',
            text_size=124,
            text=v2_text
        )

        consumer.on_page_started()
        consumer.on_revision_processed(page, rev_1)
        consumer.on_revision_processed(page, rev_2)

        df = pd.read_csv(self.csv_file)
        self.assertEqual(df.shape[0], 1)

        row = df.iloc[0]
        self.assertEqual(row['old_text'],
                         'The Scottish driver said that he had planned to perform doughnuts for the crowd, a celebration discouraged in Formula One at the time.Coulthard left Formula One after 15 years with 246 race starts and 13 wins.Red Bull team principal Christian Horner said: "It\'s a great shame for David to be eliminated from his last Grand Prix at the first corner, but he can look back on a long and illustrious career where he\'s achieved a great deal."Coulthard continued to work for Red Bull Racing in 2009 as a testing and development consultant.')
        self.assertEqual(row['new_text'],
                         'The Scottish driver said that he had planned to perform doughnuts for the crowd, a celebration discouraged in Formula One at the time.Coulthard left Formula One after 15 years with 246 race starts and 13 wins.Red Bull team principal Christian Horner said: "It\'s a great shame for David to be eliminated from his last Grand Prix at the first corner, but he can look back on a long and illustrious career where he\'s achieved a great deal."Coulthard stopped to work for Red Bull Racing forever.')

    def load_text(self, file_name):
        def read_file(v):
            with open(Path(__file__).parent / 'resources/diffs' / v / f'{file_name}.txt', 'r', encoding='utf-8') as f:
                return f.read()

        v1_text = read_file('v1')
        v2_text = read_file('v2')
        return v1_text, v2_text


if __name__ == '__main__':
    unittest.main()
