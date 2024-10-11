import csv
import multiprocessing
from dataclasses import asdict, dataclass

from wikiprefs.diffs.model import TextDiff
from wikiprefs.utils.xml_utils import Revision

logger = multiprocessing.get_logger()

DIFF_CSV_FIELDNAMES = [
    'page_title',
    'page_id',
    'rev_id',
    'prev_rev_id',
    'timestamp',
    'contributor',
    'comment',
    'section',
    'old_text',
    'new_text',
]


@dataclass
class DiffRow:
    """Single diff for given revision compared to the previous revision"""

    page_title: str
    page_id: int

    rev_id: int
    prev_rev_id: int
    timestamp: str
    contributor: str
    comment: str

    section: str = None
    old_text: str = None
    new_text: str = None


@dataclass
class Seq:
    """Sequence for aggregating new and old text during diff creation"""

    old_text: str = ''
    new_text: str = ''

    any_text_just_removed: bool = False
    any_text_removed: bool = False
    any_text_just_added: bool = False
    any_text_added: bool = False


class DiffSaver:
    """Save diffs into CSV, and handles aggregating adjacent changes in a single paragraph"""

    def __init__(self, writer: csv.DictWriter, base_row: DiffRow, revision: Revision, save_all_diffs: bool):
        """Initialize Diff saver

        Args:
            writer (csv.DictWriter): CSV writer
            base_row (DiffRow): DiffRow instance saving as base for saving rows in CSV
            revision: revision for which the diffs were created
            save_all_diffs: if all diffs should be saved. If false following diffs won't be saved:
                * diffs that are not marked as retained in next_diff method
                * diffs that only removed text
        """
        self.writer = writer
        self.base_row = base_row
        self.revision = revision
        self.save_all_diffs = save_all_diffs
        self.diffs_count = 0

        self.curr_diffs_seq = Seq()
        self.curr_plain_text = []

    def next_text(self, text: str):
        """Next string (i.e. text that's the same in both revision) in diff sequence"""
        if text == '\n\n':
            # if the text is paragraph break flush the current diff sequence since there'll be nothing more to append
            self._save_diff_and_reset()
            return

        if '\n\n' in text:
            # the text contains a paragraph break
            # flush the current diff sequence
            prefix, suffix = self._split_on_paragraph_break(text)
            if self.curr_diffs_seq.any_text_just_added and not self.curr_diffs_seq.any_text_removed:
                # if there are only changes in the new_text, append the suffix for more context
                self.curr_diffs_seq.old_text += prefix
                self.curr_diffs_seq.new_text += prefix
            self._save_diff_and_reset()

            # save beginning of the new paragraph
            if suffix:
                self.curr_plain_text.append(suffix)
            return

        self.curr_plain_text.append(text)

    def next_diff(self, diff: TextDiff, is_retained: bool):
        """Next diff (i.e. text that was changed) in diff sequence"""
        if not is_retained and not self.save_all_diffs:
            # edited text is not retained (e.g. text is not present in the latest revision)
            # AND we don't want to save all detected diffs (e.g. when this revision is not improving FA text)
            # -> ignore the diff, flush the current diff sequence since there'll be nothing more to append
            self._save_diff_and_reset()
            return

        trigger_flush = False
        old_text = diff.v1
        new_text = diff.v2

        if old_text and '\n\n' in old_text:
            old_text, _ = self._split_on_paragraph_break(old_text)
            trigger_flush = True
        if new_text and '\n\n' in new_text:
            trigger_flush = True

        # save prefix that we've aggregated so far
        plain_text = ''.join(self.curr_plain_text)
        self.curr_plain_text = []

        self.curr_diffs_seq.old_text += plain_text
        if old_text:
            self.curr_diffs_seq.old_text += old_text
            self.curr_diffs_seq.any_text_removed = True
        else:
            self.curr_diffs_seq.any_text_just_added = True

        self.curr_diffs_seq.new_text += plain_text
        if new_text:
            self.curr_diffs_seq.new_text += new_text
            self.curr_diffs_seq.any_text_added = True
        else:
            self.curr_diffs_seq.any_text_just_removed = True

        if trigger_flush:
            self._save_diff_and_reset()

    def finish(self):
        """Finish processing diffs"""
        self._save_diff_and_reset()

    def _save_diff_and_reset(self):
        old_text = self.curr_diffs_seq.old_text.strip()
        new_text = self.curr_diffs_seq.new_text.strip()
        any_text_just_removed = self.curr_diffs_seq.any_text_just_removed
        any_text_added = self.curr_diffs_seq.any_text_added
        # reset state
        self.curr_diffs_seq = Seq()
        self.curr_plain_text = []

        if not old_text or not new_text:
            return
        if any_text_just_removed and not any_text_added:
            # text was just removed in this revision (i.e. no text added or modified)
            if not self.save_all_diffs:
                return  # don't save diff that only removed text

            # if old text is longer than 2*length of the new text, then more than half of the text was removed
            # while it might be valid removal, it's probably not very useful to keep such change
            if len(old_text) > 2 * len(new_text):
                return

        row = self.base_row
        row.old_text = old_text
        row.new_text = new_text
        self.writer.writerow(asdict(row))
        self.diffs_count += 1

    def _split_on_paragraph_break(self, text: str) -> tuple[str, str]:
        r"""Split the text on paragraph break (\n\n), but keep adjacent quote or math"""
        split = text.split('\n\n')
        if len(split) == 2:
            prefix = split[0]
            suffix = split[1]

            if (
                suffix.lstrip().startswith('"')
                or suffix.lstrip().startswith("'")
                or suffix.lstrip().startswith('<math>')
            ):
                return text, ''

            return prefix, suffix

        # split_len > 2
        prefix = split[0]
        mid = split[1]
        suffix = split[-1]

        if mid.lstrip().startswith('"') or mid.lstrip().startswith("'") or mid.lstrip().startswith('<math>'):
            prefix = prefix + mid
        return prefix, suffix
