import re
from abc import ABCMeta, abstractmethod
from polo2 import PoloDb
import sqlalchemy as sa

class PoloSource(PoloDb):

    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.slug = config.ini['DEFAULT']['slug']
        dbfile = '{}-corpus.db'.format(self.slug)
        PoloDb.__init__(self, dbfile)

    @abstractmethod
    def import_src_docs(self):
        """ This method should import some source data into the src_doc table
        of the corpus database. Actually may not need this. """
        raise NotImplementedError("Please Implement this method")

    # fixme: Move this to PoloText and replace with pandas version
    def clean_text(self,text):
        text = re.sub(r'_', 'MYUNDERSCORE', text) 	# Save underscores
        text = re.sub(r'\n+',' ',text)          	# Remove newlines
        text = re.sub(r'<[^>]+>',' ',text)      	# Remove tags
        text = re.sub(r'&[^; ]+;',' ',text)     	# Remove entities
        text = re.sub(r'\W+',' ',text)          	# Remove non-letters
        text = re.sub(r'[0-9]+',' ',text)       	# Remove numbers
        #text = re.sub(r' \S ',' ',text)         	# ???
        text = re.sub(r'\s+',' ',text)          	# Collapse spaces
        text = re.sub('MYUNDERSCORE', '_', text) 	# Put back underscores
        text = text.lower()
        return text


