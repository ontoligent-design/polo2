import re, nltk
#from nltk.corpus import stopwords
#from gensim.models import TfidfModel
#from gensim.models import LdaModel
#from gensim.corpora import Dictionary
#from gensim.models.phrases import Phrases
from PoloDb import PoloDb
import pandas as pd

class PoloCorpus(PoloDb):

    NOALPHA = re.compile(r'\W+')
    MSPACES = re.compile(r'\s+')
    use_stopwords = True
    use_nltk = True

    def __init__(self, config):

        self.corpus_file = config.ini['DEFAULT']['mallet_corpus_input']
        self.corpus_sep = config.ini['DEFAULT']['corpus_sep']
        self.nltk_data_path = config.ini['DEFAULT']['nltk_data_path']
        self.slug = config.ini['DEFAULT']['slug']
        self.extra_stops = config.ini['DEFAULT']['extra_stops']

        # Local overrides of defaults
        for key in ['use_nltk', 'use_stopwords']:
            if key in config.ini['DEFAULT']:
                setattr(self, 'use_nltk', config.ini['DEFAULT'][key])

        dbfile = '{}-corpus.db'.format(self.slug)
        PoloDb.__init__(self, dbfile)
        if self.nltk_data_path: nltk.data.path.append(self.nltk_data_path)

    def import_table_stopword(self, use_nltk=False):
        swset = set()
        if use_nltk:
            from nltk.corpus import stopwords
            nltk_stopwords = set(stopwords.words('english')) # Lang needs param
            swset.update(nltk_stopwords)
        with open(self.extra_stops, 'r') as src:
            swset.update([word for word in src.read().split()])
        swdf = pd.DataFrame({'token_str': list(swset)})
        self.put_table(swdf, 'stopword')

    def import_table_doc(self):
        if self.corpus_sep == '': self.corpus_sep = ','
        doc = pd.read_csv(self.corpus_file, header=None, sep=self.corpus_sep)
        doc.columns = ['doc_id', 'doc_label', 'doc_content']
        self.put_table(doc, 'doc')

    def add_tables_doctoken_and_token(self):
        doc = self.get_table('doc')
        doc = doc[doc.doc_content.notnull()]

        doctoken = pd.concat([pd.Series(row[0], row[2].split()) for _, row in doc.iterrows()]).reset_index()
        doctoken.columns = ['token_str', 'doc_id']
        doctoken = doctoken[['doc_id', 'token_str']]

        token = pd.DataFrame(doctoken.token_str.value_counts())
        token.columns = ['token_count']

        if self.use_stopwords:
            stopwords = self.get_table('stopword')
            doctoken = doctoken[~doctoken.token_str.isin(stopwords.token_str)]
            token = token[~token.index.isin(stopwords.token_str)]

        self.put_table(doctoken, 'doctoken')
        self.put_table(token, 'token', index=True, index_label='token_str')

    def add_tables_ngram_and_docngram(self, n = 2):
        if n not in range(2, 5):
            print("n not in range")
            return(None)
        doctoken = self.get_table('doctoken')
        cols = {}
        for i in range(n):
            pad = [None] * i
            cols[str(i)] = doctoken.token_str[i:].tolist() + pad
            cols[str(n+i)] = doctoken.doc_id[i:].tolist() + pad
        docngram = pd.DataFrame(cols)
        c1 = str(n)
        c2 = str((2 * n) - 1)
        docngram = docngram[docngram[c1] == docngram[c2]]
        docngram['ngram'] = docngram.apply(lambda row: '_'.join(row[:n]), axis=1)
        docngram = docngram[[c1, 'ngram']]
        docngram.columns = ['doc_id', 'ngram']
        ngram = pd.DataFrame(docngram.ngram.value_counts())
        ngram.columns = ['ngram_count']

        prefixes = ['no', 'uni', 'bi', 'tri', 'quadri']
        self.put_table(docngram, 'doc{}gram'.format(prefixes[n]))
        self.put_table(ngram, '{}gram'.format(prefixes[n]), index=True, index_label='ngram')


if __name__ == '__main__':
    print(1)
