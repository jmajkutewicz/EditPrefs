import multiprocessing
import xml.sax
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = multiprocessing.get_logger()


class WIKI_XML:
    """Constants used int Wikipedia XML dump"""

    NS = 'http://www.mediawiki.org/xml/export-0.10/'

    PAGE = 'page'
    REVISION = 'revision'

    ID = 'id'
    TITLE = 'title'

    REV_PARENT_ID = 'parentid'
    REV_TIMESTAMP = 'timestamp'
    REV_MINOR = 'minor'
    REV_COMMENT = 'comment'
    REV_SHA1 = 'sha1'
    REV_TEXT = 'text'

    REV_CONTRIBUTOR = 'contributor'
    REV_CONTRIBUTOR_USERNAME = 'username'
    REV_CONTRIBUTOR_IP = 'ip'


@dataclass
class Page:
    """Wikipedia page"""

    title: str = ''
    id: str = ''


@dataclass
class Revision:
    """Wikipedia page revision"""

    id: str = ''
    parent_id: str = ''
    timestamp: str = ''
    is_minor: bool = False
    contributor: str = ''
    comment: str = ''
    sha1: str = ''
    text_size: int = -1
    text: [str] = field(default_factory=list, repr=False)


class PageHandler(xml.sax.ContentHandler):
    """XML handler for <page> element and it's content (except for <revision>)"""

    def __init__(self):
        self.page = Page()
        self._current_handler: Callable[[str], None] | None = None

    def startElement(self, name, attrs):
        if name == WIKI_XML.TITLE:
            self._current_handler = self._title
        elif name == WIKI_XML.ID:
            self._current_handler = self._id

    def endElement(self, name):
        if name == WIKI_XML.ID:
            self._current_handler = None
            logger.debug(f'Page id: {self.page.id}')
        elif name == WIKI_XML.TITLE:
            self._current_handler = None
            logger.debug(f'Page title: {self.page.title}')

    def characters(self, content):
        if self._current_handler is not None:
            self._current_handler(content)

    def _title(self, content: str):
        self.page.title += content

    def _id(self, content: str):
        self.page.id += content


class RevisionHandler(xml.sax.ContentHandler):
    """XML handler for <revision> element and it's content"""

    # End tags (i.e. </tag>), which doesn't need special handling of element end
    END_ELEMENTS = {
        WIKI_XML.ID,
        WIKI_XML.REV_PARENT_ID,
        WIKI_XML.REV_TIMESTAMP,
        WIKI_XML.REV_COMMENT,
        WIKI_XML.REV_SHA1,
        WIKI_XML.REV_TEXT,
    }

    def __init__(self):
        self.revision = Revision()
        self._current_handler: Callable[[str], None] | None = None
        self._in_contributor = False

    def startElement(self, name, attrs):
        if self._in_contributor:
            # contributor might be a user (then we only need username, and ignore id), or ip
            if name == WIKI_XML.REV_CONTRIBUTOR_USERNAME or name == WIKI_XML.REV_CONTRIBUTOR_IP:
                self._current_handler = self._contributor_username
            return

        if name == WIKI_XML.ID:
            self._current_handler = self._id
        elif name == WIKI_XML.REV_PARENT_ID:
            self._current_handler = self._parent_id
        elif name == WIKI_XML.REV_TIMESTAMP:
            self._current_handler = self._timestamp
        elif name == WIKI_XML.REV_COMMENT:
            self._current_handler = self._comment
        elif name == WIKI_XML.REV_SHA1:
            self._current_handler = self._sha1
        elif name == WIKI_XML.REV_TEXT:
            self.revision.text_size = self._parse_size(attrs['bytes'])
            self._current_handler = self._text
        elif name == WIKI_XML.REV_MINOR:
            self.revision.is_minor = True
        elif name == WIKI_XML.REV_CONTRIBUTOR:
            self._in_contributor = True

    def endElement(self, name):
        if self._in_contributor:
            if name == WIKI_XML.REV_CONTRIBUTOR:
                self._in_contributor = False
            elif name == WIKI_XML.REV_CONTRIBUTOR_USERNAME or name == WIKI_XML.REV_CONTRIBUTOR_IP:
                self._current_handler = None
        elif name in RevisionHandler.END_ELEMENTS:
            self._current_handler = None

    def characters(self, content):
        if self._current_handler is not None:
            self._current_handler(content)

    def _id(self, content: str):
        self.revision.id += content

    def _parent_id(self, content: str):
        self.revision.parent_id += content

    def _timestamp(self, content: str):
        self.revision.timestamp += content

    def _comment(self, content: str):
        self.revision.comment += content

    def _sha1(self, content: str):
        self.revision.sha1 += content

    def _text(self, content: str):
        self.revision.text.append(content)

    def _contributor_username(self, content: str):
        self.revision.contributor += content

    @staticmethod
    def _parse_size(s):
        try:
            return int(s)
        except ValueError:
            return -1


class PageHistoryHandler(xml.sax.ContentHandler):
    """XML handler for page meta history .xml file"""

    class _HandlerState(Enum):
        NONE = 1
        PAGE = 2
        REVISION = 3

    def __init__(self, revision_consumer: Callable[[Page, Revision], None]):
        self._state = PageHistoryHandler._HandlerState.NONE
        self._revision_consumer = revision_consumer
        self._page_handler = PageHandler()
        self._revision_handler = None

    def startElement(self, name, attrs):
        match self._state:
            case PageHistoryHandler._HandlerState.NONE:
                # start of the xml file
                if name == WIKI_XML.PAGE:
                    self._state = PageHistoryHandler._HandlerState.PAGE
                else:
                    logger.warning(f'Unexpected element: {name}')
            case PageHistoryHandler._HandlerState.PAGE:
                # start of <page> (only 1 page pare .xml file is expected)
                if name == WIKI_XML.REVISION:
                    # start of revision (they are ordered from oldest to newest in meta history xml)
                    self._state = PageHistoryHandler._HandlerState.REVISION
                    self._revision_handler = RevisionHandler()
                else:
                    # other page data (e.g. id or title)
                    self._page_handler.startElement(name, attrs)
            case PageHistoryHandler._HandlerState.REVISION:
                # elements inside <revision>
                self._revision_handler.startElement(name, attrs)

    def endElement(self, name):
        match self._state:
            case PageHistoryHandler._HandlerState.NONE:
                # there should be no elements outside <page>...</page>
                logger.warning(f'Unexpected end element: {name}')
            case PageHistoryHandler._HandlerState.PAGE:
                if name == WIKI_XML.PAGE:
                    # page end (</page>). This should be the end of .xml file
                    logger.debug(f'Finished parsing xml page {self._page_handler.page.title}')
                    self._state = PageHistoryHandler._HandlerState.NONE
                else:
                    self._page_handler.endElement(name)
            case PageHistoryHandler._HandlerState.REVISION:
                if name == WIKI_XML.REVISION:
                    # end of single revision, we're back on the page level
                    self._revision_consumer(self._page_handler.page, self._revision_handler.revision)
                    self._state = PageHistoryHandler._HandlerState.PAGE
                    self._revision_handler = None
                else:
                    self._revision_handler.endElement(name)

    def characters(self, content):
        match self._state:
            case PageHistoryHandler._HandlerState.NONE:
                pass
            case PageHistoryHandler._HandlerState.PAGE:
                self._page_handler.characters(content)
            case PageHistoryHandler._HandlerState.REVISION:
                self._revision_handler.characters(content)


class PageHistoryConsumerInterface:
    """Interface for PageHistoryConsumer"""

    def on_page_started(self) -> None:
        """Start new page processing"""
        pass

    def on_page_processed(self) -> None:
        """Finalize page processing"""
        pass

    def on_revision_processed(self, page: Page, revision: Revision) -> None:
        """Finalize revision processing"""
        pass


class MetaHistoryXmlHandler(xml.sax.ContentHandler):
    """XML content handler, can process meta-history XML dump"""

    class _HandlerState(Enum):
        NONE = 1
        PAGE = 2

    def __init__(self, page_history_consumer: PageHistoryConsumerInterface):
        self._state = MetaHistoryXmlHandler._HandlerState.NONE
        self._page_history_consumer = page_history_consumer
        self._page_history_handler = None

    def startElement(self, name, attrs):
        match self._state:
            case MetaHistoryXmlHandler._HandlerState.NONE:
                # start of the xml file
                if name == WIKI_XML.PAGE:
                    logger.debug('Starting page processing')
                    self._state = MetaHistoryXmlHandler._HandlerState.PAGE
                    self._page_history_consumer.on_page_started()

                    self._page_history_handler = PageHistoryHandler(
                        lambda page, rev: self._page_history_consumer.on_revision_processed(page, rev)
                    )
                    self._page_history_handler.startElement(name, attrs)
            case MetaHistoryXmlHandler._HandlerState.PAGE:
                self._page_history_handler.startElement(name, attrs)

    def endElement(self, name):
        match self._state:
            case MetaHistoryXmlHandler._HandlerState.NONE:
                pass
            case MetaHistoryXmlHandler._HandlerState.PAGE:
                if name == WIKI_XML.PAGE:
                    self._page_history_handler.endElement(name)
                    self._page_history_consumer.on_page_processed()

                    self._page_history_handler = None
                    self._state = MetaHistoryXmlHandler._HandlerState.NONE
                else:
                    self._page_history_handler.endElement(name)

    def characters(self, content):
        match self._state:
            case MetaHistoryXmlHandler._HandlerState.NONE:
                pass
            case MetaHistoryXmlHandler._HandlerState.PAGE:
                self._page_history_handler.characters(content)
