import os
import pandas as pd
from pandas.api.types import is_string_dtype

import numpy as np
from numpy import log

from scipy.linalg import norm
from scipy.stats import entropy

import nltk
import nltk.data
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from textblob import TextBlob # Consider changing
from gensim.models import word2vec

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron

from polo2 import PoloDb
from polo2 import PoloFile
from polo2 import PoloConfig


class PoloCorpus(PoloDb):

    ngram_prefixes = ['no', 'uni', 'bi', 'tri', 'quadri']

    def __init__(self, config:PoloConfig):
        """Initialize corpus object"""

        # Import Configs
        self.config = config
        self.config.set_config_attributes(self)
        if not os.path.isfile(self.cfg_src_file_name):
            raise ValueError("Missing source file. Check value of `src_file_name` in INI file.")
        self.dbfile = config.generate_corpus_db_file_path()
        PoloDb.__init__(self, self.dbfile)

        # self.db = PoloDb(self.dbfile) # Why not do this?

        if self.cfg_nltk_data_path: 
            nltk.data.path.append(self.cfg_nltk_data_path)

        # For tokenizing into sentences
        # fixme: TOKENIZER ASSUMES ENGLISH -- PARAMETIZE THIS
        nltk.download('punkt')
        nltk.download('tagsets')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        self.tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

    def import_table_doc(self, src_file_name=None, normalize=True):
        """Import source file into doc table"""

        # Read in file content
        if not src_file_name:
            src_file_name = self.cfg_src_file_name
        doc = pd.read_csv(src_file_name, header=0, sep=self.cfg_src_file_sep, lineterminator='\n')
        doc.index.name = 'doc_id'

        # todo: Find a more efficient way of handling this -- such as not duplicating!
        # This is a legacy of an older procedure which now has performance implications.
        if 'doc_original' not in doc.columns:
            doc['doc_original'] = doc.doc_content

        # todo: Put this in a separate and configurable function for general text normalization.
        # Preliminary normalization of documents
        doc['doc_content'] = doc.doc_content.str.replace(r'\n+', ' ', regex=True)  # Remove newlines
        doc['doc_content'] = doc.doc_content.str.replace(r'<[^>]+>', ' ', regex=True)  # Remove tags
        doc['doc_content'] = doc.doc_content.str.replace(r'\s+', ' ', regex=True)  # Collapse spaces

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
            nltk_stopwords = set(stopwords.words('english'))
            swset.update(nltk_stopwords)
        if self.cfg_extra_stops and os.path.isfile(self.cfg_extra_stops):
            src = PoloFile(self.cfg_extra_stops)
            swset.update([word for word in src.read_bigline().split()])
        swdf = pd.DataFrame({'token_str': list(swset)})
        self.put_table(swdf, 'stopword')

    # todo: Consider changing table name to DOCTERM or TOKEN
    def add_table_doctoken(self):
        """Create doctoken and doctokenbow tables; update doc table"""

        docs = self.get_table('doc', set_index=True)

        # todo: Consider dividing this in two parts, the first to create a Phrase model with Gensim
        # This takes a long time
        doctokens = pd.DataFrame([(sentences[0], j, k, token[0], token[1])
            for sentences in docs.apply(lambda x: (x.name, sent_tokenize(x.doc_content)), 1)
            for j, sentence in enumerate(sentences[1])
            for k, token in enumerate(nltk.pos_tag(nltk.word_tokenize(sentence)))],
                columns=['doc_id', 'sentence_id', 'token_ord', 'token_str', 'pos'])
        doctokens = doctokens.set_index(['doc_id', 'sentence_id', 'token_ord'])

        # Normalize
        doctokens.token_str = doctokens.token_str.str.lower()
        doctokens.token_str = doctokens.token_str.str.replace(r'[^a-z]+', '', regex=True)
        doctokens = doctokens[~doctokens.token_str.str.match(r'^\s*$')]

        # todo: Instead of removing stopwords, identify with feature
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

    # fixme: TOKEN should be the TERM table (aka VOCAB)
    def add_table_token(self):
        """Get token data from doctoken and doctokenbow"""
        doctoken = self.get_table('doctoken')
        token = pd.DataFrame(doctoken.token_str.value_counts())
        token.sort_index(inplace=True)
        token.reset_index(inplace=True)
        token.columns = ['token_str', 'token_count']
        token.index.name = 'token_id'

        # Add pos_max to token
        pos_max = doctoken.groupby(['token_str', 'pos']).pos.count().unstack().idxmax(1)
        token['pos_max'] = token.token_str.map(pos_max)

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


    # fixme: Use a better sentiment detector
    def _get_sentiment(self, doc):
        doc2 = TextBlob(doc)
        return doc2.sentiment

    def add_tfidf_to_doctokenbow(self):
        """Add TFIDF data to doctokenbow table"""

        doctokenbow = self.get_table('doctokenbow', set_index=True)
        tokens = self.get_table('token', set_index=True)
        docs = pd.read_sql_query("SELECT doc_id, token_count FROM doc", self.conn, index_col='doc_id')
        num_docs = docs.index.size

        # Compute local and gloabl token (actually term) significance

        self.alpha = .4
        doc_max = doctokenbow.groupby('doc_id').token_count.max()
        tokens['df'] = doctokenbow.groupby('token_id').token_count.count() # THIS SHOULD ALREADY BE IN VOCAB
        # n_docs = len(doctokenbow.index.levels[0])
        tokens['idf'] = np.log2(num_docs/tokens.df)
        tokens['dfidf'] = tokens.df * tokens.idf
        doctokenbow['tf'] = self.alpha + (1 - self.alpha) * (doctokenbow.token_count / doc_max)
        doctokenbow['tfidf'] = doctokenbow.tf * tokens.idf
        # doctokenbow['tfidf_l2'] = pd.DataFrame(normalize(doctokenbow['tfidf']), 
        #                                        columns=doctokenbow['tfidf'].columns, 
        #                                        index=doctokenbow['tfidf'].index)
        doctokenbow['tfidf_l2'] = doctokenbow['tfidf'] / doctokenbow.groupby(['doc_id']).apply(lambda x: norm(x.tfidf, 2))
        tokens['tfidf_sum'] = doctokenbow.groupby('token_id').tfidf_l2.sum()
        tokens['tfidf_avg'] = doctokenbow.groupby('token_id').tfidf_l2.mean()

        self.put_table(doctokenbow, 'doctokenbow', if_exists='replace', index=True)
        self.put_table(tokens, 'token', if_exists='replace', index=True)

    def add_stems_to_token(self):
        """Add stems to token table"""

        # We only use one stemmer since stemmers suck anyway :-)
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

    def add_tables_ngram_and_docngram(self, n=2):
        """Create ngram and docngram tables for n (using stems)"""
        key = {2:'bi', 3:'tri'}
        try:
            infix = key[n]
        except KeyError as e:
            print('Invalid ngram length. Must be 2 or 3')
            return False

        sql = {}
        sql['bi'] = """
        SELECT dt_x.doc_id AS doc_id, 
            t_x.token_stem_porter AS tx, 
            t_y.token_stem_porter AS ty, 
            t_x.token_stem_porter || '_' || t_y.token_stem_porter AS ngram, 
            count() AS tf
        FROM doctoken dt_x 
        JOIN doctoken dt_y ON (dt_x.doc_id = dt_y.doc_id 
            AND dt_x.sentence_id = dt_y.sentence_id 
            AND dt_y.rowid = (dt_x.rowid + 1))
        JOIN token t_x ON dt_x.token_str = t_x.token_str
        JOIN token t_y ON dt_y.token_str = t_y.token_str
        GROUP BY dt_x.doc_id, ngram
        """
        sql['tri'] = """
        SELECT dt_x.doc_id AS doc_id, 
            t_x.token_stem_porter AS tx, 
            t_y.token_stem_porter AS ty, 
            t_z.token_stem_porter AS tz,
            t_x.token_stem_porter || '_' || t_y.token_stem_porter || '_' || t_z.token_stem_porter AS ngram, 
            count() AS tf
        FROM doctoken dt_x 
        JOIN doctoken dt_y ON (dt_x.doc_id = dt_y.doc_id 
            AND dt_x.sentence_id = dt_y.sentence_id 
            AND dt_y.rowid = (dt_x.rowid + 1))
        JOIN doctoken dt_z ON (dt_x.doc_id = dt_z.doc_id 
            AND dt_x.sentence_id = dt_z.sentence_id 
            AND dt_z.rowid = (dt_y.rowid + 1)) 
        JOIN token t_x ON dt_x.token_str = t_x.token_str
        JOIN token t_y ON dt_y.token_str = t_y.token_str            
        JOIN token t_z ON dt_z.token_str = t_z.token_str            
        GROUP BY dt_x.doc_id, ngram
        """

        docngrams = pd.read_sql(sql[infix], self.conn)
        self.put_table(docngrams, 'ngram{}doc'.format(infix), index=False)

    def add_stats_to_ngrams(self, type='bi'):
        """Create distinct ngram tables with stats"""

        sql1 = """
        SELECT g.doc_id, d.doc_label, g.ngram, g.tf
        FROM ngram{}doc g
        JOIN doc d USING(doc_id)        
        """.format(type)

        sql2 = """
        SELECT ngram, doc_label, sum(tf) AS tf_sum 
        FROM (
            SELECT g.doc_id, d.doc_label, g.ngram, g.tf
            FROM ngram{}doc g
            JOIN doc d USING(doc_id)
        )
        GROUP BY doc_label, ngram
        """.format(type)

        sql3 = """
        WITH stats(n) AS (SELECT COUNT() AS n FROM doc)
        SELECT ngram, count() AS c, (SELECT n FROM stats) AS n, 
        CAST(COUNT() AS REAL) / CAST((SELECT n FROM stats) AS REAL) AS df
        FROM ngram{}doc
        GROUP BY ngram
        ORDER BY c DESC
        """.format(type)

        docngram = pd.read_sql_query(sql1, self.conn, index_col='doc_id')
        labelngram = pd.read_sql_query(sql2, self.conn,  index_col=['ngram','doc_label'])
        ngramstats = pd.read_sql_query(sql3, self.conn,  index_col=['ngram'])

        ndm = labelngram.unstack().fillna(0)
        ndm.columns = ndm.columns.droplevel(0)

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
        # ngram['df'] = docngram.groupby(['ngram']).doc_id.count()
        ngram['idf'] = np.log10(1 / ngramstats.df) # ngram.apply(lambda x: np.log10(N/x.df), 1)
        ngram['freq'] = f_t
        ngram['entropy'] = h
        ngram['score'] = score

        self.put_table(ngram, 'ngram{}'.format(type), index=True)
        self.put_table(ndm, 'ngram{}doc_group_matrix'.format(type), index=True)

    def save_ngrams_as_replacements(self, type='bi'):
        """Creatte replacement files from ngram tables"""
    
        table = "ngram{}".format(type)
        sql1 = "SELECT AVG(idf), AVG(freq), AVG(entropy) FROM {}".format(table)
        params = pd.read_sql_query(sql1, self.conn).iloc[0].tolist()
    
        sql2 = """
        SELECT REPLACE(n.ngram, '_', ' ') FROM {} n
        WHERE n.idf < ? AND n.freq > ? AND n.entropy > ? 
	    ORDER BY FREQ DESC LIMIT 5000
        """.format(table)

        replacements = pd.read_sql_query(sql2, self.conn, params=params)
        outfilename = 'corpus/replacements_{}.txt'.format(table)
        replacements.to_csv(outfilename, index=False, header=False)

    def add_tables_ngram_and_docngram_old(self, n = 2):
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
                            & (docngram['sentence_id'] == docngram['sentence_id'+suffix])]
                            # Remove ngrams that cross sentences

        # Remove redundant doc and sentence cols
        cols = ['doc_id', 'sentence_id', 'token_str'] + \
               ['token_str_{}'.format(i) for i in range(1, n)]
        docngram = docngram[cols]

        # Join the grams into a single ngram
        docngram.set_index(['doc_id', 'sentence_id'], inplace=True)
        #docngram['ngram'] = docngram.apply(lambda x: x.str.cat(sep='_'), axis=1)  # SLOW!
        #docngram['ngram'] = docngram.apply('_'.join, axis=1) # Faster, but still slow
        docngram['ngram'] = docngram.sum(1)  # FAST, but above we had to suffix our terms
        docngram = docngram[['ngram']]

        self.put_table(docngram, 'ngram{}doc'.format(self.ngram_prefixes[n]), index=True)

        # fixme: Create fuction here
        # Get ngrams sorted by special sauce
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

    def add_word2vec_table(self, k=246, window=5,
                        min_count=100, workers=4,
                        perplexity=40, n_components=2,
                        init='pca', n_iter=2500, random_state=23):
        """Use Gensim to generate word2vec model from doctoken table"""
        
        doctoken = self.get_table('doctoken', set_index=True)
        corpus = doctoken.groupby('doc_id').apply(lambda x: x.token_str.tolist()).values.tolist()
        model = word2vec.Word2Vec(corpus,
                                  vector_size=k, window=window,
                                  min_count=min_count, workers=workers)
        del corpus
        df = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)
        del model
        df.columns = ['F{}'.format(i) for i in range(k)]
        df.index.name = 'token_str'  # todo: Later change this to term_str
        
        tsne_model = TSNE(perplexity=perplexity,
                          n_components=n_components, init=init,
                          n_iter=n_iter, random_state=random_state)
        tsne_values = tsne_model.fit_transform(df)
        df['tsne_x'] = tsne_values[:, 0]
        df['tsne_y'] = tsne_values[:, 1]

        self.put_table(df, 'word_embedding', index=True)

    def add_pca_tables(self, k_components=10, n_terms=1000):
        """Create a PCA table with k components, n terms from doctokenbow table"""
        
        sql_vocab = """
        SELECT token_id, token_str 
        FROM token 
        WHERE pos_max NOT LIKE 'NNP%'
        ORDER BY tfidf_sum DESC 
        LIMIT ?
        """

        sql_bow = """
        SELECT doc_id, token_id, token_str, tfidf_l2 as tfidf 
        FROM doctokenbow 
        JOIN token USING (token_id)
        WHERE token_id IN (
            SELECT token_id 
            FROM token
            WHERE pos_max NOT LIKE 'NNP%'   
            ORDER BY tfidf_sum DESC 
            LIMIT ?            
        ) 
        """
        # note: Should parametize exclusion of NNP    

        doctokenbow = pd.read_sql_query(sql_bow, self.conn, params=(n_terms,), index_col=['doc_id', 'token_id'])
        vocab = pd.read_sql_query(sql_vocab, self.conn, params=(n_terms,), index_col='token_id')
        docs = doctokenbow['tfidf'].unstack().fillna(0)

        print(vocab)

        pca = PCA(n_components=k_components)
        # projected = pca.fit_transform(normalize(docs.values, norm='l2'))
        projected = pca.fit_transform(docs.values) # Normalized already
        pccols = ["PC{}".format(i) for i in range(k_components)]
        pca_doc = pd.DataFrame(projected, columns=pccols, index=docs.index)
        pca_term = pd.DataFrame((pca.components_.T * np.sqrt(pca.explained_variance_)),
                                columns=pccols, index=docs.columns)
        pca_term['token_str'] = vocab['token_str']
        pca_item = self._get_pca_terms(pca_term, k_components)
        pca_term = pca_term.drop('token_str', axis=1)
        pca_item['explained_variance'] = pd.DataFrame(pca.explained_variance_)
        pca_item['explained_variance_ratio'] = pd.DataFrame(pca.explained_variance_ratio_)

        pca_doc_narrow = pca_doc.stack().reset_index()
        pca_doc_narrow.columns = ['doc_id', 'pc_id', 'pc_weight']
        pca_doc_narrow['pc_id'] = pca_doc_narrow['pc_id'].apply(lambda x: x.replace('PC', '')).astype('int')
        pca_doc_narrow = pca_doc_narrow.set_index(['doc_id', 'pc_id']).dropna().sort_index()

        pca_term_narrow = pca_term.stack().reset_index()
        pca_term_narrow.columns = ['token_id', 'pc_id', 'pc_weight']
        pca_term_narrow['pc_id'] = pca_term_narrow['pc_id'].apply(lambda x: x.replace('PC', '')).astype('int')
        pca_term_narrow = pca_term_narrow.set_index(['pc_id', 'token_id']).dropna().sort_index()

        # todo: Decide if you want to keep narrow and wide tables
        self.put_table(pca_item, 'pca_item', index=True)
        self.put_table(pca_doc, 'pca_doc', index=True, index_label='doc_id')
        self.put_table(pca_term, 'pca_term', index=True)
        self.put_table(pca_term_narrow, 'pca_term_narrow', index=True)
        self.put_table(pca_doc_narrow, 'pca_doc_narrow', index=True)

    def _get_pca_terms(self, V, k=10, n=15):
        loadings = []
        for i in range(k):
            PC = f'PC{i}'
            cols = ['token_str', PC]
            pos = V.sort_values(PC, ascending=False).iloc[:n][cols].copy()
            neg = V.sort_values(PC, ascending=True).iloc[:n][cols].copy()
            pos['pc_id'] = i
            neg['pc_id'] = i
            neg['pc_val'] = neg[PC]
            pos['val'] = 'pos'
            neg['val'] = 'neg'
            pos['pc_val'] = pos[PC]
            loadings.append(pos)
            loadings.append(neg)
        loadings = pd.concat(loadings, sort=True).iloc[:, k:].reset_index(drop=True).set_index('pc_id')
        loadings = loadings.groupby(['pc_id', 'val']).token_str.apply(lambda x: ' '.join(x)) \
            .to_frame().unstack()
        loadings.columns = loadings.columns.droplevel(0)

        print(loadings)
        return loadings

    def get_perceptron_models(self, max_v=4000):
        """Run a perceptron classifier using labels"""
        
        sql1 = """
        SELECT doc_id, token_str, token_id, tfidf_l2 as tfidf 
        FROM doctokenbow as bow
        JOIN (SELECT token_id, token_str 
            FROM token 
            WHERE token_str NOT IN (SELECT token_str FROM stopword) 
                ORDER BY tfidf_sum 
                DESC LIMIT ?) as vocab 
            USING (token_id)
        """
        bow = pd.read_sql_query(sql1, self.conn, params=(max_v,),
            index_col=['doc_id', 'token_id'])
        vocab = bow.reset_index()[['token_id', 'token_str']]\
            .drop_duplicates().set_index('token_id')    
        dtm = bow['tfidf'].unstack(fill_value=0)

        sql2 = "SELECT doc_id, doc_label FROM doc"
        docs = pd.read_sql_query(sql2, self.conn, index_col='doc_id')
    
        X = dtm.values
        y = docs.loc[dtm.index, 'doc_label'].fillna('NO_LABEL').values

        # pd.DataFrame(X).to_csv('X.csv')
        # pd.DataFrame(y).to_csv('y.csv')
        
        model = Perceptron(tol=1e-3, random_state=0)
        model.fit(X, y)
        weights = pd.DataFrame(model.coef_, 
            index=model.classes_, 
            columns=vocab.loc[dtm.columns].token_str.values).T
        self.put_table(weights,'perceptron_weights', index=True, index_label='token_str')
    

    # def add_nnmf_tables(self, k=20):
    #     from sklearn.preprocessing import normalize
    #     from sklearn.decomposition import NMF
    #
    #     doctokenbow = self.get_table('doctokenbow', set_index=True)
    #     vocab = self.get_table('token', set_index=True)
    #     stops = self.get_table('stopword')
    #     vocab_reduced = vocab[~vocab.token_str.isin(stops)] \
    #         .doc_count.sort_values(ascending=False).head(v_terms)
    #     doctokenbow = doctokenbow.loc[vocab_reduced.index]
    #     tfidf = doctokenbow['tfidf'].unstack().fillna(0)
    #     vocab_idx = tfidf.columns
    #     doc_idx = tfidf.index

    def export_mallet_corpus(self):
        """Create a MALLET corpus file"""
        # token_type = 'token_str'  # Could also be token_stem_porter token_stem_snowball
        # We export the doctoken table as the input corpus to MALLET. This preserves our normalization
        # between the corpus and trial model databases.

        # mallet_corpus_sql = """
        # CREATE VIEW mallet_corpus AS
        # SELECT dt.doc_id, d.doc_label, GROUP_CONCAT({}, ' ') AS doc_content
        # FROM doctoken dt JOIN doc d USING (doc_id) JOIN token t USING (token_str)
        # GROUP BY dt.doc_id
        # ORDER BY dt.doc_id
        # """.format(token_type)
        # self.conn.execute("DROP VIEW IF EXISTS mallet_corpus")
        # self.conn.execute(mallet_corpus_sql)
        # self.conn.commit()
        # mallet_corpus = pd.read_sql_query('SELECT * FROM mallet_corpus', self.conn)

        # REPLACED token_str WITH token_atem_porter
        # mallet_corpus_sql = """
        # SELECT dt.doc_id, d.doc_label, GROUP_CONCAT(t.token_stem_porter, ' ') AS doc_content
        # FROM doctoken dt 
        # JOIN doc d USING (doc_id) 
        # JOIN token t USING (token_str)
        # GROUP BY dt.doc_id
        # ORDER BY dt.doc_id
        # """

        # No stemming; only nouns ...
        mallet_corpus_sql = """
        SELECT dt.doc_id, d.doc_label, GROUP_CONCAT(dt.token_str , ' ') AS doc_content
        FROM doctoken dt JOIN doc d USING (doc_id)
        WHERE dt.pos LIKE "NN%"
        GROUP BY dt.doc_id
        ORDER BY dt.doc_id        
        """
        mallet_corpus = pd.read_sql_query(mallet_corpus_sql, self.conn)

        if is_string_dtype(mallet_corpus['doc_label']):
            mallet_corpus['doc_label'] = mallet_corpus['doc_label'].str.replace(r'\s+', '_', regex=True)
            
        mallet_corpus.to_csv(self.cfg_mallet_corpus_input, index=False, header=False, sep=',')