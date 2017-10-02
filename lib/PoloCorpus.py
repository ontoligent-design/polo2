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
        # fixme: Is this the best way to handle config overrides? If so, add to other classes
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
        # todo: Craete file rw function to wrap error trapping, etc.
        if self.corpus_sep == '': self.corpus_sep = ','
        doc = pd.read_csv(self.corpus_file, header=None, sep=self.corpus_sep)
        doc.columns = ['doc_key', 'doc_label', 'doc_content']
        doc = doc.set_index(['doc_key'])
        # fixme: Reconcile this with what mallet is doing!
        # fixme: Put this in a separate function for general text manipulation
        # fixme: Create mallet corpus from doc table and turn off its stopwards
        # todo: Consider providing orderdicts of replacements that users can choose or create
        doc = doc[doc.doc_content.notnull()]
        #doc['doc_original'] = doc.doc_content
        doc['doc_content'] = doc.doc_content.str.lower()
        doc['doc_content'] = doc.doc_content.str.replace(r'_', 'MYUNDERSCORE') # Keep underscores
        doc['doc_content'] = doc.doc_content.str.replace(r'\n+', ' ') # Remove newlines
        doc['doc_content'] = doc.doc_content.str.replace(r'<[^>]+>', ' ') # Remove tags
        doc['doc_content'] = doc.doc_content.str.replace(r'\W+', ' ') # Remove non-alphanumerics
        doc['doc_content'] = doc.doc_content.str.replace(r'[0-9]+', ' ') # Remove numbers
        doc['doc_content'] = doc.doc_content.str.replace(r'\s+', ' ') # Collapse spaces
        doc['doc_content'] = doc.doc_content.str.replace('MYUNDERSCORE', '_') # Put underscores back
        self.put_table(doc, 'doc', index=True)

    def add_table_doctoken(self):
        doc = self.get_table('doc')
        doctoken = pd.concat([pd.Series(row[0], row[2].split()) for _, row in doc.iterrows()]).reset_index()
        doctoken.columns = ['token_str', 'doc_key']
        doctoken = doctoken[['doc_key', 'token_str']]
        if self.use_stopwords:
            stopwords = self.get_table('stopword')
            doctoken = doctoken[~doctoken.token_str.isin(stopwords.token_str)]
        self.put_table(doctoken, 'doctoken')

    def add_table_token(self):
        doctoken = self.get_table('doctoken')
        token = pd.DataFrame(doctoken.token_str.value_counts())
        token.index.name = 'token_str'
        token.columns = ['token_count']
        self.put_table(token, 'token', index=True)

    def add_table_doctokenbow(self):
        doctoken = self.get_table('doctoken')
        doctokenbow = pd.DataFrame(doctoken.groupby('doc_key').token_str.value_counts())
        doctokenbow.columns = ['token_count']
        self.put_table(doctokenbow, 'doctokenbow', index=True)

    ngram_prefixes = ['no', 'uni', 'bi', 'tri', 'quadri']
    def add_tables_ngram_and_docngram(self, n = 2):
        if n not in range(2, 5):
            print("n not in range")
            return(None)
        doctoken = self.get_table('doctoken')
        cols = {}
        for i in range(n):
            pad = [None] * i
            cols[str(i)] = doctoken.token_str[i:].tolist() + pad
            cols[str(n+i)] = doctoken.doc_key[i:].tolist() + pad
        docngram = pd.DataFrame(cols)
        c1 = str(n)
        c2 = str((2 * n) - 1)
        docngram = docngram[docngram[c1] == docngram[c2]]
        docngram['ngram'] = docngram.apply(lambda row: '_'.join(row[:n]), axis=1)
        docngram = docngram[[c1, 'ngram']]
        docngram.columns = ['doc_key', 'ngram']

        ngram = pd.DataFrame(docngram.ngram.value_counts())
        ngram.index.name = 'ngram'
        ngram.columns = ['ngram_count']

        self.put_table(docngram, 'ngram{}doc'.format(self.ngram_prefixes[n]))
        self.put_table(ngram, 'ngram{}'.format(self.ngram_prefixes[n]), index=True)


if __name__ == '__main__':
    print(1)
