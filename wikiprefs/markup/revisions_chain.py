from __future__ import annotations

import logging
import multiprocessing

from wikiprefs.utils.xml_utils import Revision

logger = multiprocessing.get_logger()


class RevisionNode:
    """Single revision chained with its parent"""

    def __init__(self, idx: str, comment: str, timestamp: str, sha1: str, text_size: int, parent: RevisionNode):
        """Initialize the revision node"""
        self.id = idx
        self.comment = comment
        self.timestamp = timestamp
        self.sha1 = sha1
        self.text_size = text_size

        self.parent = parent


class RevisionsChainCreator:
    """Creates revisions chain"""

    def __init__(self):
        """Initialize the revision chain creator"""
        self.sha1_to_rev = {}
        self.all_revisions = []

        self.reverts_count = 0
        self.revisions_count = 0
        self.curr_revision: RevisionNode | None = None

    def on_page_started(self) -> None:
        """Start processing new page"""
        self.sha1_to_rev = {}
        self.all_revisions = []

        self.reverts_count = 0
        self.revisions_count = 0
        self.curr_revision = None

    def on_page_processed(self) -> list[RevisionNode]:
        """Finished processing page"""
        logger.info(f'Reverts: {self.reverts_count} / {self.revisions_count}')

        revisions = []
        rev_chain_length = 1
        revision = self.curr_revision
        while revision is not None:
            revisions.append(revision)
            rev_chain_length += 1

            revision = revision.parent
        logger.info(f'Revision chain length {rev_chain_length}')

        revisions.reverse()
        return revisions

    def on_revision_processed(self, revision: Revision) -> None:
        """Process next revision"""
        self.all_revisions.append(revision.id)
        self.revisions_count += 1

        # use both text sha1 hash and size to decreases chances of getting a collision on just sha1
        key = (revision.sha1, revision.text_size)
        if self.curr_revision and key == (self.curr_revision.sha1, self.curr_revision.text_size):
            # no changes in text
            pass
        if key not in self.sha1_to_rev:
            parent = self.curr_revision
            self.curr_revision = RevisionNode(
                revision.id, revision.comment, revision.timestamp, revision.sha1, revision.text_size, parent
            )
            self.sha1_to_rev[key] = self.curr_revision
        else:
            revert_parent = self.sha1_to_rev[key]
            self.curr_revision = revert_parent
            self.reverts_count += 1

            if logger.isEnabledFor(logging.DEBUG):
                parent_revision = self.curr_revision
                logger.debug(
                    f'Revision {revision.id}[{revision.timestamp}] reverts to '
                    f'{parent_revision.id}[{parent_revision.timestamp}]'
                )

                revert_count = 1
                while self.all_revisions[-revert_count] != revert_parent.id:
                    revert_count += 1
                logger.debug(f'\t{revert_count} reverted edits')
