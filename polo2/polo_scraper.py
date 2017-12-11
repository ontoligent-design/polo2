from abc import ABCMeta, abstractmethod
from polo2 import PoloConfig

class PoloScraper:

    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.config = config
        self.corpus_file = self.config.ini['DEFAULT']['src_file_name']
        self.corpus_file_sep = self.config.ini['DEFAULT']['src_file_sep']

    @abstractmethod
    def make_corpus(self):
        """This method must output to disk a file in Standard Corpus File Format. This method
        will be called by posh to refresh the corpus."""
        pass
