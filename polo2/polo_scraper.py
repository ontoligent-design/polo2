from abc import ABCMeta, abstractmethod
from polo2 import PoloConfig

class PoloScraper:

    __metaclass__ = ABCMeta

    @abstractmethod
    def load_config(self, config_file):
        self.config = PoloConfig(config_file)
        self.corpus_file = self.config.ini['DEFAULT']['src_file_name']
        self.corpus_file_sep = self.config.ini['DEFAULT']['src_file_sep']

    @abstractmethod
    def make_corpus(self):
        pass
