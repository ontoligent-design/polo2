from polo2 import PoloConfig
from polo2 import PoloDb

class PoloGensim:

    def __init__(self, conifg, corpus_dbfile = None):
        self.gs_corpus = None
        self.gs_dict = None
        self.db = PoloDb(corpus_dbfile)

    def make_gs_corpus(self):
        self.gs_corpus = []
        doc = self.db.get_table('doc')

    def make_gs_dict(self):
        self.gs_dict = {}