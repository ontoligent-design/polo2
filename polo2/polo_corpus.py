import os
import re
import nltk
import nltk.data
from textblob import TextBlob
import pandas as pd
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
        stopwords = self.get_table('stopword').token_str.tolist()
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        sentence_id = 0

        # Handle replacements -- EXPERIMENTAL
        reps = []
        rfile_name = '{}/{}'.format(self.cfg_base_path, self.cfg_replacements)
        if os.path.exists(rfile_name):
            rfile = PoloFile(rfile_name)
            reps = [tuple(line.strip().split('|')) for line in rfile.read_lines()]
        else:
            print(rfile_name, 'not found')

        # Not efficient but intelligible
        doctoken = dict(doc_id=[], sentence_id=[], token_str=[], token_pos=[])
        for doc_id in docs.index:
            doc = docs.loc[doc_id].doc_content
            for sentence_raw in tokenizer.tokenize(doc):

                # Normalize
                sentence = sentence_raw.lower()
                sentence = re.sub(r'[^\w\s]','', sentence)

                # Do replacements
                for rep in reps:
                    sentence = re.sub(rep[0], rep[1], sentence)

                sentence_id += 1
                # todo: Parametize using pos, stopwords, removing numbers
                for token_str, token_pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
                    if token_str not in stopwords and not re.match(r'^\d+$', token_str):
                        doctoken['doc_id'].append(doc_id)
                        doctoken['sentence_id'].append(sentence_id)
                        doctoken['token_str'].append(token_str)
                        doctoken['token_pos'].append(token_pos)

        doctoken_df = pd.DataFrame(doctoken)
        self.put_table(doctoken_df, 'doctoken', if_exists='replace', index=False)

        # Creates a BOW model for the doc, removing words in sequence and only keeping counts
        doctokenbow = pd.DataFrame(doctoken_df.groupby('doc_id').token_str.value_counts())
        doctokenbow.columns = ['token_count']
        self.put_table(doctokenbow, 'doctokenbow', index=True)

        # Add token counts to doc
        docs['token_count'] = doctokenbow.groupby('doc_id').token_count.sum()
        self.put_table(docs, 'doc', if_exists='replace', index=True)

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
        docngram_cols = {}
        for i in range(n):
            pad = [None] * i
            docngram_cols[str(i)] = doctoken.token_str[i:].tolist() + pad
            docngram_cols[str(n+i)] = doctoken.doc_id[i:].tolist() + pad
            docngram_cols[str(n*2 + i)] = doctoken.sentence_id[i:].tolist() + pad
        docngram = pd.DataFrame(docngram_cols)
        c1 = str(n)
        c2 = str(2*n - 1)
        c3 = str(2*n)
        c4 = str(3*n - 1)
        docngram = docngram[(docngram[c1] == docngram[c2]) & (docngram[c3] == docngram[c4])]
        docngram['ngram'] = docngram.apply(lambda row: '_'.join(row[:n]), axis=1)
        docngram = docngram[[c1, c3, 'ngram']]
        docngram.columns = ['doc_id', 'sentence_id', 'ngram']
        self.put_table(docngram, 'ngram{}doc'.format(self.ngram_prefixes[n]))

        ngram = pd.DataFrame(docngram.ngram.value_counts())
        ngram.index.name = 'ngram'
        ngram.columns = ['ngram_count']
        self.put_table(ngram, 'ngram{}'.format(self.ngram_prefixes[n]), index=True)

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
        token_type = 'token_str' # token_stem_porter token_stem_snowball
        
        _mallet_corpus_sql = """
        CREATE VIEW mallet_corpus AS
        SELECT dt.doc_id, d.doc_label, GROUP_CONCAT(ngram, ' ') AS doc_content
        FROM ngrambidoc dt JOIN doc d USING (doc_id) JOIN ngrambi t USING (ngram)
        GROUP BY dt.doc_id
        ORDER BY dt.doc_id
        """

        mallet_corpus_sql = """
        CREATE VIEW mallet_corpus AS
        SELECT dt.doc_id, d.doc_label, GROUP_CONCAT({}, ' ') AS doc_content
        FROM doctoken dt JOIN doc d USING (doc_id) JOIN token t USING (token_str)
        WHERE dt.token_pos LIKE 'NN%' OR dt.token_pos LIKE 'VB%' OR dt.token_pos LIKE 'JJ%'
        GROUP BY dt.doc_id
        ORDER BY dt.doc_id
        """.format(token_type)
        self.conn.execute("DROP VIEW IF EXISTS mallet_corpus")
        self.conn.execute(mallet_corpus_sql)
        self.conn.commit()
        mallet_corpus = pd.read_sql_query('SELECT * FROM mallet_corpus', self.conn)
        mallet_corpus.to_csv(self.cfg_mallet_corpus_input, index=False, header=False, sep=',')