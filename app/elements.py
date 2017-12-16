from polo2 import PoloDb
import pandas as pd
from scipy import stats

class Elements(object):

    def __init__(self, config, trial_name='trial1'):

        # Set some values
        if trial_name not in config.trials:
            raise ValueError("Invalid trail name `{}`.format(trial)")
        self.config = config
        self.trial = trial_name
        self.slug = self.config.ini['DEFAULT']['slug']
        self.base_path = self.config.ini['DEFAULT']['base_path']
        self.thresh = float(self.config.ini['DEFAULT']['thresh'])

        # Load the databases
        corpus_db_file = self.config.generate_corpus_db_file_path()
        model_db_file = self.config.generate_model_db_file_path(self.trial)
        self.corpus = PoloDb(corpus_db_file)
        self.model = PoloDb(model_db_file)

    def get_doc_count(self):
        self.doc_count = pd.read_sql_query('SELECT count(*) AS n FROM doc', self.corpus.conn).n.tolist()[0]
        return self.doc_count

    def get_topic_count(self):
        self.topic_count = pd.read_sql_query('SELECT count(*) AS n FROM topic', self.model.conn).n.tolist()[0]
        return self.topic_count

    def get_topic(self, topic_id):
        topic_id = int(topic_id)
        sql = 'SELECT * FROM topic WHERE topic_id = {}'.format(topic_id)
        df = pd.read_sql_query(sql, self.model.conn)
        df.set_index('topic_id', inplace=True)
        df['topic_phrases'] = self.get_topic_phrases(topic_id)
        return df

    def get_top_bigrams(self, limit = 50):
        limit = int(limit)
        sql = "SELECT ngram, ngram_count from ngrambi ORDER BY ngram_count DESC LIMIT {}".format(limit)
        df = pd.read_sql_query(sql, self.corpus.conn)
        df['ngram_percent'] = (df.ngram_count / df.ngram_count.max() * 100).astype('int')
        return df

    def get_topic_phrases(self, topic_id):
        topic_id = int(topic_id)
        sql = "SELECT topic_phrase FROM topicphrase WHERE topic_id = {} ORDER BY phrase_weight DESC".format(topic_id)
        phrases = ', '.join(pd.read_sql_query(sql, self.model.conn).topic_phrase.tolist())
        return phrases

    def get_topic_entropy_hist(self):
        doctopics = self.model.get_table('doctopic', set_index=True)
        doctopics.unstack()

    def get_topicdoclabel_matrix(self, sort_by_alpha = True):
        dtm = self.model.get_table('topicdoclabel_matrix', set_index=False)
        col1 = dtm.columns.tolist()[0]
        dtm.set_index(col1, inplace=True)

        topics = self.model.get_table('topic', set_index=True)
        if sort_by_alpha:
            topics = topics.sort_values('topic_alpha', ascending=True)
        dtm = dtm[topics.index.astype('str').tolist()]
        dtm.columns = topics.reset_index().apply(lambda x: 'T{} {}'.format(x.topic_id, x.topic_words), axis=1)

        return dtm

    def get_topicdocord_matrix(self):
        dtm = self.model.get_table('topicdocord_matrix', set_index=False)
        col1 = dtm.columns.tolist()[0]
        dtm.set_index(col1, inplace=True)
        return dtm

    def get_topicdoc_ord_for_topic(self, topic_id):
        topic_id = int(topic_id)
        doc_col = self.config.ini['DEFAULT']['src_ord_col']
        sql = "SELECT {0} as ord_val, `{1}` as topic_weight FROM topicdocord_matrix ORDER BY {0}".format(doc_col, topic_id)
        df = pd.read_sql_query(sql, self.model.conn)
        return df

    def get_docs_for_topic_and_label(self, topic_id, doc_col_value, doc_col = None):
        if not doc_col:
            doc_col = self.config.ini['DEFAULT']['src_ord_col'] # Should wrap these calls with a method
        src_docs = pd.read_sql_query("SELECT * FROM doc "
                                     "WHERE {} = ? LIMIT 100".format(doc_col), self.corpus.conn, params=(doc_col_value,))
        return src_docs

    def get_doc_entropy(self):
        sql = "SELECT ROUND(topic_entropy, 2) as h FROM doc"
        df = pd.read_sql_query(sql, self.model.conn)
        return df

    def test(self):
        return 1


