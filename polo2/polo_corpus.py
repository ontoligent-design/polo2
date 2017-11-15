import os, sys
import nltk
import pandas as pd

from polo2 import PoloDb
from polo2 import PoloFile

class PoloCorpus(PoloDb):

    use_stopwords = True
    use_nltk = True
    ngram_prefixes = ['no', 'uni', 'bi', 'tri', 'quadri']

    def __init__(self, config):

        self.slug = config.ini['DEFAULT']['slug']
        self.corpus_file = config.ini['DEFAULT']['mallet_corpus_input']
        self.nltk_data_path = config.ini['DEFAULT']['nltk_data_path']
        self.extra_stops = config.ini['DEFAULT']['extra_stops']
        self.normalize = config.ini['DEFAULT']['normalize']

        # Source file stuff
        self.src_file_name = config.ini['DEFAULT']['src_file_name']
        if not os.path.isfile(self.src_file_name):
            raise ValueError("Missing source file. Check value of `src_file_name` in INI file.")
        self.src_file_sep = config.ini['DEFAULT']['src_file_sep']
        self.src_base_url = config.ini['DEFAULT']['src_base_url']
        self.src_ord_col = config.ini['DEFAULT']['src_ord_col']

        # Local overrides of defaults
        for key in ['use_nltk', 'use_stopwords']:
            if key in config.ini['DEFAULT']:
                setattr(self, 'use_nltk', config.ini['DEFAULT'][key])

        self.dbfile = '{}-corpus.db'.format(self.slug)
        PoloDb.__init__(self, self.dbfile)
        if self.nltk_data_path: nltk.data.path.append(self.nltk_data_path)

    # todo: Add normalization key to config.ini and respond accordingly
    def import_table_doc(self, src_file_name=None, normalize=True):
        # todo: Clarify requirements for doc -- delimitter, columns, header, etc.
        # All of this stuff should be in a schema as you did before
        if not src_file_name: src_file_name = self.src_file_name
        if self.src_file_sep == '':
            self.src_file_sep = '|'
        elif self.src_file_sep == 'TAB':
            self.src_file_sep = '\t'
        doc = pd.read_csv(src_file_name, header=0, sep=self.src_file_sep)
        # fixme: Reconcile this with what mallet is doing!
        # fixme: Put this in a separate function for general text manipulation
        # fixme: Create mallet corpus from doc table and turn off its stopwards
        # todo: Consider providing orderdicts of replacements that users can choose or create
        if int(self.normalize) == 1:
            doc['doc_content'] = doc.doc_content.str.lower()
            #doc['doc_content'] = doc.doc_content.str.replace(r'_', 'MYUNDERSCORE') # Keep underscores
            doc['doc_content'] = doc.doc_content.str.replace(r'\n+', ' ') # Remove newlines
            doc['doc_content'] = doc.doc_content.str.replace(r'<[^>]+>', ' ') # Remove tags
            doc['doc_content'] = doc.doc_content.str.replace(r'\W+', ' ') # Remove non-alphanumerics
            doc['doc_content'] = doc.doc_content.str.replace(r'[0-9]+', ' ') # Remove numbers
            doc['doc_content'] = doc.doc_content.str.replace(r'\s+', ' ') # Collapse spaces
            #doc['doc_content'] = doc.doc_content.str.replace('MYUNDERSCORE', '_') # Put underscores back
        doc.index.name = 'doc_id'
        self.put_table(doc, 'doc', index=True)

    def import_table_stopword(self, use_nltk=False):
        swset = set()
        if use_nltk:
            from nltk.corpus import stopwords
            nltk_stopwords = set(stopwords.words('english')) # Lang needs param
            swset.update(nltk_stopwords)
        if self.extra_stops and os.path.isfile(self.extra_stops):
            src = PoloFile(self.extra_stops)
            swset.update([word for word in src.read_bigline().split()])
        swdf = pd.DataFrame({'token_str': list(swset)})
        self.put_table(swdf, 'stopword')

    def add_table_doctoken(self):
        # todo: Add token_ord column
        doc = self.get_table('doc')
        doctoken = pd.concat([pd.Series(row.doc_id, row.doc_content.split()) for _, row in doc.iterrows()]).reset_index()
        doctoken.columns = ['token_str', 'doc_id']
        doctoken = doctoken[['doc_id', 'token_str']]
        if self.use_stopwords:
            stopwords = self.get_table('stopword')
            doctoken = doctoken[~doctoken.token_str.isin(stopwords.token_str)]
        doctokenbow = pd.DataFrame(doctoken.groupby('doc_id').token_str.value_counts())
        doctokenbow.columns = ['token_count']
        self.put_table(doctokenbow, 'doctokenbow', index=True)
        self.put_table(doctoken, 'doctoken')

    def add_table_token(self):
        doctoken = self.get_table('doctoken')
        token = pd.DataFrame(doctoken.token_str.value_counts())
        token.sort_index(inplace=True)
        token.reset_index(inplace=True)
        token.columns = ['token_str', 'token_count']
        token.index.name = 'token_id'
        self.put_table(token, 'token', index=True)

        token.reset_index(inplace=True)
        doctokenbow = self.get_table('doctokenbow')
        doctokenbow = doctokenbow.merge(token[['token_id', 'token_str']], on="token_str")
        doctokenbow = doctokenbow[['doc_id', 'token_id', 'token_count']]
        doctokenbow.sort_values('doc_id', inplace=True)
        self.put_table(doctokenbow, 'doctokenbow', if_exists='replace')

    def add_tables_ngram_and_docngram(self, n = 2):
        if n not in range(2, 5):
            raise ValueError("n not in range. Must be between 2 and 4 inclusive.")
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
        ngram.index.name = 'ngram'
        ngram.columns = ['ngram_count']
        self.put_table(docngram, 'ngram{}doc'.format(self.ngram_prefixes[n]))
        self.put_table(ngram, 'ngram{}'.format(self.ngram_prefixes[n]), index=True)

    def export_mallet_corpus(self):
        polo_corpus = self.get_table('doc')
        #polo_corpus = polo_corpus[['doc_id', 'doc_label', 'doc_content']]
        polo_corpus = polo_corpus[['doc_label', 'doc_content']] # Why does get_table() pull doc_id as index?
        polo_corpus.to_csv(self.corpus_file, index=True, index_label='doc_id', header=False)
