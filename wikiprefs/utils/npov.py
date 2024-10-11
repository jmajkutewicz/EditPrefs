from dataclasses import dataclass

NPOV_FIELDNAMES = ['page_id', 'page_title', 'rev_id', 'parent_rev_id', 'comment']


@dataclass
class NpovEdit:
    """NPOV edit revision metadata"""

    page_id: str
    page_title: str
    rev_id: str
    parent_rev_id: str
    comment: str
