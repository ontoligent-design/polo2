import re, nltk
#from nltk.corpus import stopwords
#from gensim.models import TfidfModel
#from gensim.models import LdaModel
#from gensim.corpora import Dictionary
#from gensim.models.phrases import Phrases
from PoloDb import PoloDb
import pandas as pd
import numpy as np

class PoloCorpus(PoloDb):

    NOALPHA = re.compile(r'\W+')
    MSPACES = re.compile(r'\s+')

    def __init__(self, config):

        self.corpus_file = config.ini['DEFAULT']['mallet_corpus_input']
        self.corpus_sep = config.ini['DEFAULT']['corpus_sep']
        self.nltk_data_path = config.ini['DEFAULT']['nltk_data_path']
        self.slug = config.ini['DEFAULT']['slug']

        dbfile = '{}-corpus.db'.format(self.slug)
        PoloDb.__init__(self, dbfile)
        if self.nltk_data_path: nltk.data.path.append(self.nltk_data_path)

    def import_table_doc(self):
        if self.corpus_sep == '': self.corpus_sep = ','
        doc = pd.read_csv(self.corpus_file, header=None, sep=self.corpus_sep)
        doc.columns = ['doc_id', 'doc_label', 'doc_content']
        self.df_to_db(doc, 'doc')

    def add_tables_doctoken_and_token(self):
        doc = self.db_to_df('doc')
        doc = doc[doc.doc_content.notnull()]

        doctoken = pd.concat([pd.Series(row[0], row[2].split()) for _, row in doc.iterrows()]).reset_index()
        doctoken.columns = ['token_str', 'doc_id']
        doctoken = doctoken[['doc_id', 'token_str']]

        token = pd.DataFrame(doctoken.token_str.value_counts())
        token.columns = ['token_count']

        self.df_to_db(doctoken, 'doctoken')
        self.df_to_db(token, 'token', index=True, index_label='token_str')

    def add_table_bigram(self):
        pass

    def get_table(self, table_name = None):
        return self.db_to_df(table_name)

    def get_ngrams(self, n = 2):
        doctoken = self.db_to_df('doctoken')
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

        prefixes = ['no', 'uni', 'bi', 'tri']
        self.df_to_db(docngram, 'doc{}gram'.format(prefixes[n]))
        self.df_to_db(ngram, '{}gram'.format(prefixes[n]), index=True, index_label='ngram')

        '''
        doctokens = [
            #[token for token in nltk.word_tokenize(doc[2]) if token not in self.stopwords]
            token for token in nltk.word_tokenize(src_doc[src_doc.doc_id == doc_id]['doc_content'])
            for doc_id in src_doc.doc_id
        ]
        '''
        """
        if len(self.docs) == 0: self.pull_src_docs_as_docs()
        # Create so-called TEXT (doctokens) from processed DOCUMENT lines
        doctokens = [
            #[token for token in nltk.word_tokenize(doc[2]) if token not in self.stopwords]
            [token for token in nltk.word_tokenize(doc[2])]
            for doc in self.docs
        ]
        # Find bigrams and update so-called TEXT
        # For a faster implementation, use the gensim.models.phrases.Phraser class
        # Also note taht Mallet has a replacements file too
        bigrams = Phrases(doctokens)
        self.doctokens = list(bigrams[doctokens])
        """
    def produce(self):
        pass
        """
        with self.dbi as db:

            # Get so-called DOCUMENT (list of lines)
            self.pull_src_docs_as_docs()
            ndocs = len(self.docs)

            # Create so-called TEXT (doctokens) from processed DOCUMENT lines
            self.generate_doctokens()

            # Create Gensim DICTIONARY from TEXT
            dictionary = Dictionary(self.doctokens)

            # Create Gensim CORPUS of DOCVECs
            corpus = [[] for _ in range(ndocs)]
            for doc_idx in range(ndocs):
                corpus[doc_idx] = dictionary.doc2bow(self.doctokens[doc_idx])

            # Create TFIDF model
            tfidf = TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]

            # Serialize things to database
            # NOTE: Replace this with a serialization class at some point
            db.create_table('word')
            for word_id, word_str in dictionary.iteritems():
                db.cur.execute('INSERT INTO word (word_id, word_str) VALUES (?,?)',(word_id,word_str))
            db.create_table('doc')
            db.create_table('docword')
            for doc_idx, doc in enumerate(self.docs):
                db.cur.execute('INSERT INTO doc (doc_index,doc_id,doc_label,doc_str) VALUES (?,?,?,?)',(doc_idx,doc[0],doc[1],doc[2]))
                doc_id = doc[0]
                for  a, b in zip(corpus[doc_idx],corpus_tfidf[doc_idx]):
                    #if a[0] != b[0]: print("Houston, we have a problem ...")
                    word_id = a[0]
                    word_count = a[1]
                    word_str = dictionary.get(word_id)
                    tfidf_weight = b[1] # placeholder
                    db.cur.execute('INSERT INTO docword (doc_index,doc_id,word_id,word_str,word_count,tfidf_weight) VALUES (?,?,?,?,?,?)',(doc_idx,doc_id,word_id,word_str,word_count,tfidf_weight))
        """

    def run_lda(self, z=10):
        pass
        """
        # Should check if these are in memory and pull if necessary
        self.pull_gensim_corpus()
        self.pull_gensim_id2token()

        lda = LdaModel(self.gensim_corpus, id2word=self.gensim_id2token, num_topics=z, alpha='asymmetric', eval_every=10)

        # print_topics(num_topics=20, num_words=10)
        topics = lda.print_topics(num_topics=z, num_words=10)
        #print(topics)

        # get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
        doctopic = []
        for doc in self.gensim_corpus:
            doctopic.append(lda.get_document_topics(doc))
        #print(doctopic)
        """

    def update_word_freqs(self):
        pass
        """
        with self.dbi as db:
            rows = []
            for r in db.cur.execute("SELECT sum(word_count) as 'word_freq', word_id FROM docword GROUP BY word_id"):
                rows.append(r)
            db.cur.executemany("UPDATE word SET word_freq = ? WHERE word_id = ?",rows)
        """

    def update_word_stems(self):
        pass
        """
        from nltk.stem import PorterStemmer
        st = PorterStemmer()
        with self.dbi as db:
            rows = []
            for r in db.cur.execute('SELECT word_id, word_str FROM word'):
                stem = st.stem(r[1])
                rows.append((stem,r[0]))
            db.cur.executemany("UPDATE word SET word_stem = ? WHERE word_id = ?",rows)
        """

    def pull_corpus_as_words(self):
        pass
        """
        self.doc_words = []
        with self.dbi as db:
            for r in db.cur.execute("SELECT doc_str FROM doc ORDER BY doc_index"):
                for word in r[0].split():
                    self.doc_words.append(word)
        """

    def insert_bigrams(self,n=10):
        pass
        """
        from nltk.collocations import BigramCollocationFinder
        from nltk.metrics import BigramAssocMeasures
        rows = []
        self.pull_corpus_as_words()
        stopset = set(self.stopwords)
        filter_stops = lambda w: len(w) < 3 or w in stopset
        finder = BigramCollocationFinder.from_words(self.doc_words)
        finder.apply_word_filter(filter_stops)
        finder.apply_freq_filter(3)
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        scored = finder.score_ngrams(bigram_measures.raw_freq)
        for bigram, score in scored:
            rows.append([bigram[0],bigram[1],score])
        with self.dbi as db:
            db.insert_values('bigram',rows)
        """

    def pull_gensim_corpus(self):
        pass
        """
        with self.dbi as db:
            n = db.cur.execute("SELECT COUNT(*) FROM docword").fetchone()[0]
            self.gensim_corpus = [[] for _ in range(n)]
            for r in db.cur.execute('SELECT doc_index, word_id, word_count FROM docword ORDER BY doc_id, word_id'):
                self.gensim_corpus[r[0]].append((r[1], r[2]))
        """

    def pull_gensim_token2id(self):
        pass
        """
        self.gensim_token2id = {}
        with self.dbi as db:
            for r in db.cur.execute("SELECT word_str,word_id FROM word"):
                self.gensim_token2id[r[0]] = r[1]
        """

    def pull_gensim_id2token(self):
        pass
        """
        self.gensim_id2token = {}
        with self.dbi as db:
            for r in db.cur.execute("SELECT word_str,word_id FROM word"):
                self.gensim_id2token[r[1]] = r[0]
        """

    def generate_mallet_corpus(self, outfile):
        pass
        """
        with self.dbi as db, open(outfile,'w') as out:
            docs = {}
            for r in db.cur.execute("SELECT doc_id || '', doc_label, word_str FROM doc JOIN docword USING(doc_id)"):
                doc_key = 'doc_{}'.format(r[0])
                try:
                    #docs[doc_key]['doc_id'] = r[0]
                    #docs[doc_key]['doc_label'] = r[1]
                    docs[doc_key]['doc_str'] += ' ' + r[2]
                except:
                    docs[doc_key] = {'doc_id':r[0],'doc_label':r[1],'doc_str':r[2]}
            for doc in docs:
                out.write('{},{},{}\n'.format(docs[doc]['doc_id'],docs[doc]['doc_label'],docs[doc]['doc_str']))
        """

if __name__ == '__main__':
    print(1)
