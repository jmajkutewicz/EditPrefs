import unittest

from wikiprefs.diffs.filtering import CommentFilter


class TestCommentFiltering(unittest.TestCase):
    def test_pov_filter(self):
        positive_cases = [
            'Reverted to revision 823347961 by [[Special:Contributions/*Treker|*Treker]] ([[User talk:*Treker|talk]]): See [[WP:NPOV]]. ([[WP:TW|TW]])',
            'Reverted to revision 823347961 by [[Special:Contributions/*Treker|*Treker]] ([[User talk:*Treker|talk]]): See [[WP:POV]]. ([[WP:TW|TW]])',
            'rv back - changes were made following discussion at NPOV/N.  And personally, I think they do improve the flow.',
            'rv back - changes were made following discussion at POV/N.  And personally, I think they do improve the flow.',
            'Reverted 3 edits by [[Special:Contributions/213.31.11.80|213.31.11.80]] ([[User talk:213.31.11.80|talk]]); POV edits. ([[WP:TW|TW]])',
            'rm npov statement - its not much of an achievement.',
            '/* History */ unsourced POV',
        ]
        negative_cases = [
            'described poverty',
            'Add Nanopov section',
        ]

        comments_filter = CommentFilter()

        for case in positive_cases:
            case = case.lower()
            self.assertTrue(comments_filter._is_pov(case), f'Failed to find POV in {case}')
        for case in negative_cases:
            case = case.lower()
            self.assertFalse(comments_filter._is_pov(case), f'Found false positive POV in {case}')

    def test_is_npov_edit(self):
        tc = [
            (False, 'Revert NPOV improvement'),
            (False, 'rollback rev 2: removed pov'),
            (False, 'undid WP:SPAM'),
            (False, 'per [[WP:CONTEXTBIO]]'),
            (False, '/* top */ no need to link the same thing twice ([[WP:OVERLINK]], [[WP:CONCISE]]).'),
            (True, 'improved section controvercies; neutralized bias'),
            (True, 'test (see WP:FRINGE)'),
            (True, 'more work needed to get round WP:UNDUE, though'),
        ]

        comments_filter = CommentFilter()

        for t in tc:
            is_npov = t[0]
            comment = t[1]
            with self.subTest(f'{t[1]}'):
                self.assertEqual(is_npov, comments_filter.is_npov_edit(comment))


if __name__ == '__main__':
    unittest.main()
