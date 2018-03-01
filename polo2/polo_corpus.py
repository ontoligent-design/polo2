import os, sys, re
import nltk, nltk.data
from textblob import TextBlob
import pandas as pd

from polo2 import PoloDb
from polo2 import PoloFile

class PoloCorpus(PoloDb):

    use_stopwords = True
    use_nltk = True
    ngram_prefixes = ['no', 'uni', 'bi', 'tri', 'quadri']

    def __init__(self, config):

        # todo: Have general way of ingesting config or just create a corpus config
        self.slug = config.ini['DEFAULT']['slug']
        self.corpus_file = config.ini['DEFAULT']['mallet_corpus_input']
        self.nltk_data_path = config.ini['DEFAULT']['nltk_data_path']
        self.extra_stops = config.ini['DEFAULT']['extra_stops']
        self.normalize = config.ini['DEFAULT']['normalize']
        if 'normalize' in config.ini['DEFAULT'].keys():
            self.sentiment = config.ini['DEFAULT']['sentiment'] # todo: Add sentiment to config template
        else:
            self.sentiment = 0

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

        self.dbfile = config.generate_corpus_db_file_path()
        PoloDb.__init__(self, self.dbfile)
        if self.nltk_data_path: nltk.data.path.append(self.nltk_data_path)

        # For tokenizing into sentences
        # fixme: TOKENIZER ASSUMES ENGLISH
        nltk.download('punkt')
        self.tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

    def import_table_doc(self, src_file_name=None, normalize=True):
        # todo: Clarify requirements for doc -- delimitter, columns, header, etc.
        # All of this stuff should be in a schema as you did before
        if not src_file_name:
            src_file_name = self.src_file_name
        doc = pd.read_csv(src_file_name, header=0, sep=self.src_file_sep)
        doc.index.name = 'doc_id'

        # todo: Find a more efficient way of handling this
        # todo: Get rid of doc_original throughout
        if 'doc_original' not in doc.columns:
            doc['doc_original'] = doc.doc_content

        # fixme: Put this in a separate and configurable function for general text normalization.
        """
        if int(self.normalize) == 1:
            doc['doc_content'] = doc.doc_content.str.lower()
            doc['doc_content'] = doc.doc_content.str.replace(r'_+', ' ')  # Remove underscores
            doc['doc_content'] = doc.doc_content.str.replace(r'\n+', ' ') # Remove newlines
            doc['doc_content'] = doc.doc_content.str.replace(r'<[^>]+>', ' ') # Remove tags
            doc['doc_content'] = doc.doc_content.str.replace(r'\W+', ' ') # Remove non-alphanumerics
            doc['doc_content'] = doc.doc_content.str.replace(r'\d+', ' ') # Remove numbers
            doc['doc_content'] = doc.doc_content.str.replace(r'\s+', ' ') # Collapse spaces
            doc['doc_content'] = doc.doc_content.str.replace(r'(^\s+|\s+$)', '') # Remove leading and trailing spaces
        """
        """
        if int(self.sentiment) == 1:
            doc['doc_sentiment'] = doc.doc_content.apply(self._get_sentiment)
            doc['doc_sentiment_polarity'] = doc.doc_sentiment.apply(lambda x: round(x[0], 1))
            doc['doc_sentiment_subjectivity'] = doc.doc_sentiment.apply(lambda x: round(x[1], 2))
            del(doc['doc_sentiment'])
        """
        doc = doc[~doc.doc_content.isnull()]
        self.put_table(doc, 'doc', index=True)

    def _get_sentiment(self, doc):
        doc2 = TextBlob(doc)
        return doc2.sentiment

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
        docs = self.get_table('doc')
        stopwords = self.get_table('stopword').token_str.tolist()
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        sentence_id = 0
        doctoken = dict(doc_id=[], sentence_id=[], token_str=[])
        for doc_id in docs.index:
            doc = docs.loc[doc_id].doc_content
            for sentence_raw in tokenizer.tokenize(doc):
                sentence = re.sub(r'[^\w\s]','', sentence_raw).lower()
                sentence_id += 1
                for token_str in sentence.split():
                    if token_str not in stopwords:
                        doctoken['doc_id'].append(doc_id)
                        doctoken['sentence_id'].append(sentence_id)
                        doctoken['token_str'].append(token_str)
        doctoken_df = pd.DataFrame(doctoken)
        self.put_table(doctoken_df, 'doctoken', if_exists='replace', index=False)
        doctokenbow = pd.DataFrame(doctoken_df.groupby('doc_id').token_str.value_counts())
        doctokenbow.columns = ['token_count']
        self.put_table(doctokenbow, 'doctokenbow', index=True)

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
        doctokenbow.set_index(['doc_id', 'token_id'], inplace=True)
        self.put_table(doctokenbow, 'doctokenbow', if_exists='replace', index=True)
        
    def add_sentimant_to_doc(self):
        doc = self.get_table('doc', set_index=True)
        doc['doc_sentiment'] = doc.doc_content.apply(self._get_sentiment)
        doc['doc_sentiment_polarity'] = doc.doc_sentiment.apply(lambda x: round(x[0], 1))
        doc['doc_sentiment_subjectivity'] = doc.doc_sentiment.apply(lambda x: round(x[1], 2))
        del(doc['doc_sentiment'])
        self.put_table(doc, 'doc', index=True)

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
        """We export the doctoken table as the input corpus to MALLET. This preserves our normalization
        between the corpus and trial model databases."""
        # Can't get pandas to do this, so resorting to SQL
        mallet_corpus_sql = """
        CREATE VIEW mallet_corpus AS
        SELECT dt.doc_id, d.doc_label, GROUP_CONCAT(token_str, ' ') AS doc_content
        FROM doctoken dt JOIN doc d USING (doc_id)
        GROUP BY dt.doc_id
        ORDER BY dt.doc_id
        """
        self.conn.execute("DROP VIEW IF EXISTS mallet_corpus")
        self.conn.execute(mallet_corpus_sql)
        self.conn.commit()
        mallet_corpus = pd.read_sql_query('SELECT * FROM mallet_corpus', self.conn)
        #rgx = re.compile(r'\s+')
        #mallet_corpus['doc_label'] = mallet_corpus.doc_label.str.replace(rgx, '_')
        mallet_corpus.to_csv(self.corpus_file, index=False, header=False, sep=',')

        """
        # This does not work like it does in Jupyter. It does not concatenate the texts but instead
        # outputs one word per doc :-( 
        doctokens = self.get_table('doctoken', set_index=True)
        doc = self.get_table('doc', set_index=True)
        polo_corpus = pd.DataFrame(doctokens.groupby(doctokens.index).apply(lambda x: x.token_str.str.cat(sep=' ')))
        polo_corpus.columns = ['doc_content']
        polo_corpus['doc_label'] = doc.doc_label
        polo_corpus = polo_corpus[['doc_label', 'doc_content']]
        polo_corpus.to_csv(self.corpus_file, index=True, index_label='doc_id', header=False)
        """