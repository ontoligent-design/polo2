import os
import re
import pandas as pd
import nltk
import nltk.data
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from polo2 import PoloDb
from polo2 import PoloFile


class PoloCorpus(PoloDb):

    ngram_prefixes = ['no', 'uni', 'bi', 'tri', 'quadri']

    def __init__(self, config):
        """Initialize corpus object"""

        self.config = config
        self.config.set_config_attributes(self)
        if not os.path.isfile(self.cfg_src_file_name):
            raise ValueError("Missing source file. Check value of `src_file_name` in INI file.")
        self.dbfile = config.generate_corpus_db_file_path()
        PoloDb.__init__(self, self.dbfile)
        if self.cfg_nltk_data_path: nltk.data.path.append(self.cfg_nltk_data_path)

        # For tokenizing into sentences
        # fixme: TOKENIZER ASSUMES ENGLISH -- PARAMETIZE THIS
        nltk.download('punkt')
        nltk.download('tagsets')
        nltk.download('averaged_perceptron_tagger')
        self.tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

    def import_table_doc(self, src_file_name=None, normalize=True):
        """Import source file into doc table"""
        if not src_file_name:
            src_file_name = self.cfg_src_file_name
        doc = pd.read_csv(src_file_name, header=0, sep=self.cfg_src_file_sep)
        doc.index.name = 'doc_id'

        # todo: Find a more efficient way of handling this -- such as not duplicating!
        # This is a legacy of an older procedure which now has performance implications.
        if 'doc_original' not in doc.columns:
            doc['doc_original'] = doc.doc_content

        # todo: Put this in a separate and configurable function for general text normalization.
        # Preliminary normalization of documents
        doc['doc_content'] = doc.doc_content.str.replace(r'\n+', ' ')  # Remove newlines
        doc['doc_content'] = doc.doc_content.str.replace(r'<[^>]+>', ' ')  # Remove tags
        doc['doc_content'] = doc.doc_content.str.replace(r'\s+', ' ')  # Collapse spaces

        # Remove empty docs
        doc = doc[~doc.doc_content.isnull()]
        doc.reindex()

        self.put_table(doc, 'doc', index=True)

    def import_table_stopword(self, use_nltk=False):
        """Import stopwords"""
        swset = set()
        # fixme: Cast integers in config object
        # fixme: Parametize language
        if int(self.cfg_use_nltk) == 1:
            from nltk.corpus import stopwords
            nltk_stopwords = set(stopwords.words('english'))
            swset.update(nltk_stopwords)
        if self.cfg_extra_stops and os.path.isfile(self.cfg_extra_stops):
            src = PoloFile(self.cfg_extra_stops)
            swset.update([word for word in src.read_bigline().split()])
        swdf = pd.DataFrame({'token_str': list(swset)})
        self.put_table(swdf, 'stopword')
        
    def add_table_doctoken(self):
        """Create doctoken and doctokenbow tables; update doc table"""
        docs = self.get_table('doc', set_index=True)

        # Replacements?
        # reps = self._get_replacments()

        doctokens = pd.DataFrame([(sentences[0], j, k, token)
                     for sentences in docs.apply(lambda x: (x.name, sent_tokenize(x.doc_content)), 1)
                     for j, sentence in enumerate(sentences[1])
                     for k, token in enumerate(nltk.word_tokenize(sentence))
                     ], columns=['doc_id', 'sentence_id', 'token_ord', 'token_str'])
        doctokens.set_index(['doc_id', 'sentence_id', 'token_ord'], inplace=True)

        # Normalize
        #doctokens = doctokens[doctokens.token_pos.str.match(r'^(NN|JJ|VB)')]
        doctokens.token_str = doctokens.token_str.str.lower()
        doctokens.token_str = doctokens.token_str.str.replace(r'[^a-z]+', '')
        doctokens = doctokens[~doctokens.token_str.str.match(r'^\s*$')]

        # Remove stopwords
        stopwords = self.get_table('stopword').token_str.tolist()
        doctokens = doctokens[~doctokens.token_str.isin(stopwords)]

        self.put_table(doctokens, 'doctoken', if_exists='replace', index=True)

        # Creates a BOW model for the doc, removing words in sequence and only keeping counts
        doctokenbow = pd.DataFrame(doctokens.groupby('doc_id').token_str.value_counts())
        doctokenbow.columns = ['token_count']
        self.put_table(doctokenbow, 'doctokenbow', index=True)

        # Add token counts to doc
        docs['token_count'] = doctokenbow.groupby('doc_id').token_count.sum()
        self.put_table(docs, 'doc', if_exists='replace', index=True)

    def _get_replacments(self):
        reps = []
        rfile_name = '{}/{}'.format(self.cfg_base_path, self.cfg_replacements)
        if os.path.exists(rfile_name):
            rfile = PoloFile(rfile_name)
            reps = [tuple(line.strip().split('|')) for line in rfile.read_lines()]
        else:
            print(rfile_name, 'not found')
        return reps

    def add_table_token(self):
        """Get token data from doctoken and doctokenbow"""
        doctoken = self.get_table('doctoken')
        token = pd.DataFrame(doctoken.token_str.value_counts())
        token.sort_index(inplace=True)
        token.reset_index(inplace=True)
        token.columns = ['token_str', 'token_count']
        token.index.name = 'token_id'

        # Replace token_str with token_id in doctokenbow
        token.reset_index(inplace=True)
        doctokenbow = self.get_table('doctokenbow')
        doctokenbow = doctokenbow.merge(token[['token_id', 'token_str']], on="token_str")
        doctokenbow = doctokenbow[['doc_id', 'token_id', 'token_count']]
        doctokenbow.sort_values('doc_id', inplace=True)
        doctokenbow.set_index(['doc_id', 'token_id'], inplace=True)
        self.put_table(doctokenbow, 'doctokenbow', if_exists='replace', index=True)

        # Add doc counts to token
        token.set_index('token_id', inplace=True)
        token['doc_count'] = doctokenbow.groupby('token_id').count()
        self.put_table(token, 'token', index=True)

    def _get_sentiment(self, doc):
        doc2 = TextBlob(doc)
        return doc2.sentiment

    def add_tfidf_to_doctokenbow(self):
        """Add TFIDF data to doctokenbow table"""
        # t = token
        # TFIDF(t) = TF(t) × IDF(t)
        # TF(t)  = measure of frequency of t in document
        # IDF(t) = measure of how few documents contain t
        #        = log(NumberOfDocuments/NumberOfDocumentsContaining (t))
        from numpy import log
        doctokenbow = self.get_table('doctokenbow', set_index=True)
        tokens = self.get_table('token', set_index=True)
        docs = pd.read_sql_query("SELECT doc_id, token_count FROM doc", self.conn, index_col='doc_id')
        num_docs = docs.index.size
        doctokenbow['tf'] = doctokenbow.token_count.divide(docs.token_count)
        doctokenbow['tfidf'] = doctokenbow.tf.multiply(1 + log(num_docs / tokens.doc_count))
        self.put_table(doctokenbow, 'doctokenbow', if_exists='replace', index=True)
        tokens['tfidf_sum'] = doctokenbow.groupby('token_id').tfidf.sum()
        tokens['tfidf_avg'] = doctokenbow.groupby('token_id').tfidf.mean()
        self.put_table(tokens, 'token', if_exists='replace', index=True)

    def add_stems_to_token(self):
        """Add stems to token table"""
        # We only use one stemmer since stemmers suck anyway :-)
        from nltk.stem.porter import PorterStemmer
        porter_stemmer = PorterStemmer()
        tokens = self.get_table('token', set_index=True)
        tokens['token_stem_porter'] = tokens.token_str.apply(porter_stemmer.stem)
        self.put_table(tokens, 'token', if_exists='replace', index=True)

    def add_sentimant_to_doc(self):
        """Add sentiment to doc table"""
        doc = self.get_table('doc', set_index=True)
        doc['doc_sentiment'] = doc.doc_content.apply(self._get_sentiment)
        doc['doc_sentiment_polarity'] = doc.doc_sentiment.apply(lambda x: round(x[0], 1))
        doc['doc_sentiment_subjectivity'] = doc.doc_sentiment.apply(lambda x: round(x[1], 2))
        del(doc['doc_sentiment'])
        self.put_table(doc, 'doc', index=True)

    def add_tables_ngram_and_docngram(self, n = 2):
        """Create ngram and docngram tables for n"""
        """This may seem slow, but it doesn't hang like Gensim's version"""
        if n not in range(2, 5):
            raise ValueError("n not in range. Must be between 2 and 4 inclusive.")
        doctoken = self.get_table('doctoken')

        # Build the ngram dataframe by staggered alignment of doctoken
        dfs = []
        dummy = pd.DataFrame(dict(doc_id=[None], sentence_id=[None], token_str=[None]))
        for i in range(n):
            dt = doctoken[['doc_id', 'sentence_id', 'token_str']]
            for _ in range(n - 1 - i): # Prepend dummy rows
                dt = pd.concat([dummy, dt], ignore_index=True)
            for _ in range(i): # Append dummy rows
                dt = pd.concat([dt, dummy], ignore_index=True)
            if i < n - 1: # Suffix join glue to prevent doing a join or cat below
                dt['token_str'] = dt['token_str'] + '_'
            if i > 0: # Don't prefix the first table
                dt = dt.add_suffix('_{}'.format(i))
            dfs.append(dt)
        docngram = pd.concat(dfs, axis=1)

        # Remove ngrams that cross sentences or which have padding
        suffix = '_{}'.format(n - 1)
        docngram = docngram[(docngram['doc_id'] == docngram['doc_id'+suffix])
                            & (docngram['sentence_id'] == docngram['sentence_id'+suffix])]  # Remove ngrams that cross sentences

        # Remove redundant doc and sentence cols
        cols = ['doc_id', 'sentence_id', 'token_str'] + ['token_str_{}'.format(i) for i in range(1, n)]
        docngram = docngram[cols]

        # Join the grams into a single ngram
        docngram.set_index(['doc_id', 'sentence_id'], inplace=True)
        #docngram['ngram'] = docngram.apply(lambda x: x.str.cat(sep='_'), axis=1)  # SLOW!
        #docngram['ngram'] = docngram.apply('_'.join, axis=1) # Faster, but still slow
        docngram['ngram'] = docngram.sum(1)  # FAST, but above we had to suffix our terms
        docngram = docngram[['ngram']]

        self.put_table(docngram, 'ngram{}doc'.format(self.ngram_prefixes[n]), index=True)

        #fixme: Create fuction here
        # Get ngrams sorted by special sauce
        from scipy.stats import entropy
        docs = pd.read_sql_query("SELECT doc_id, doc_label FROM doc", self.conn, index_col='doc_id')
        docngram = docngram.join(docs.doc_label)
        ndm = docngram.groupby(['doc_label', 'ngram']).ngram.count()
        ndm = ndm[ndm > 2].unstack().fillna(0).T
        f_d = ndm.sum(0)
        f_t = ndm.sum(1)
        p_d = f_d / f_d.sum()
        p_t = f_t / f_t.sum()
        p_td = ndm.div(f_d)
        p_dt = p_td.apply(lambda x: p_d.loc[x.name] / p_t.loc[x.index]) * p_td
        h = p_dt.apply(entropy, 1)
        score = (p_t * h).sort_values(ascending=False)

        ngram = pd.DataFrame(docngram.ngram.value_counts())
        ngram.index.name = 'ngram'
        ngram.columns = ['ngram_count']
        ngram['freq'] = f_t
        ngram['entropy'] = h
        ngram['score'] = score
        self.put_table(ngram, 'ngram{}'.format(self.ngram_prefixes[n]), index=True)

        # Test
        self.put_table(ndm, 'ngram{}doc_group_matrix'.format(self.ngram_prefixes[n]), index=True)

    def add_bigram_tables(self):
        """Convenience function to add ngram tables for n = 2"""
        self.add_tables_ngram_and_docngram(n=2)

    def add_trigram_tables(self):
        """Convenience function to add ngram tables for n = 3"""
        self.add_tables_ngram_and_docngram(n=3)

    def export_mallet_corpus(self):
        """Create a MALLET corpus file"""
        # We export the doctoken table as the input corpus to MALLET. This preserves our normalization
        # between the corpus and trial model databases.
        
        # _mallet_corpus_sql = """
        # CREATE VIEW mallet_corpus AS
        # SELECT dt.doc_id, d.doc_label, GROUP_CONCAT(ngram, ' ') AS doc_content
        # FROM ngrambidoc dt JOIN doc d USING (doc_id) JOIN ngrambi t USING (ngram)
        # GROUP BY dt.doc_id
        # ORDER BY dt.doc_id
        # """

        token_type = 'token_str' # Could also be token_stem_porter token_stem_snowball
        # _mallet_corpus_sql = """
        # CREATE VIEW mallet_corpus AS
        # SELECT dt.doc_id, d.doc_label, GROUP_CONCAT({}, ' ') AS doc_content
        # FROM doctoken dt JOIN doc d USING (doc_id) JOIN token t USING (token_str)
        # WHERE dt.token_pos LIKE 'NN%' OR dt.token_pos LIKE 'VB%' OR dt.token_pos LIKE 'JJ%'
        # GROUP BY dt.doc_id
        # ORDER BY dt.doc_id
        # """.format(token_type)
        
        mallet_corpus_sql = """
        CREATE VIEW mallet_corpus AS
        SELECT dt.doc_id, d.doc_label, GROUP_CONCAT({}, ' ') AS doc_content
        FROM doctoken dt JOIN doc d USING (doc_id) JOIN token t USING (token_str)
        GROUP BY dt.doc_id
        ORDER BY dt.doc_id
        """.format(token_type)

        self.conn.execute("DROP VIEW IF EXISTS mallet_corpus")
        self.conn.execute(mallet_corpus_sql)
        self.conn.commit()
        mallet_corpus = pd.read_sql_query('SELECT * FROM mallet_corpus', self.conn)
        mallet_corpus.to_csv(self.cfg_mallet_corpus_input, index=False, header=False, sep=',')