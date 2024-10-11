from dataclasses import asdict, dataclass


@dataclass
class TextDiff:
    """Difference between two strings

    Longer class information...
    Longer class information...

    Attributes:
        v1 (str): old string version, i.e. reference point
        v2 (str): new string version, i.e. improved text
    """

    v1: str | None = None
    v2: str | None = None

    def to_json(self):
        """Converts the object to json"""
        return asdict(self)


@dataclass
class SegmentDiff:
    """Difference between two revisions of the same sections segment (i.e. some adjacent lines)

    Attributes:
        diffs ([Diff | str]): a mixed list that contain:
            * plain strings for unchanged sentences
            * TextDiff objects for changed sections of the text.
                Each diffs contains grouped sentences that differ between v1 and v2
    """

    diffs: [TextDiff | str]

    def to_json(self):
        """Converts the object to json"""
        return [d if isinstance(d, str) else d.to_json() for d in self.diffs]


@dataclass
class SectionDiff:
    """All differences between two revisions of the same sections

    Attributes:
        section (str): section name
        all_segments_diffs ([[Diff | str]]): a list of Diff
                                            (each section can contain multiple diffs in different parts of the text)
    """

    section: str
    all_segments_diffs: [SegmentDiff]

    def to_json(self):
        """Converts the object to json"""
        diffs_json = [d.to_json() for d in self.all_segments_diffs]
        return {'section': self.section, 'diffs': diffs_json}
