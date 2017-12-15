from polo2 import PoloDb
import pandas as pd

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
        sql = 'SELECT * FROM topic WHERE topic_id = {}'.format(topic_id)
        df = pd.read_sql_query(sql, self.model.conn)
        df.set_index('topic_id', inplace=True)
        df['topic_phrases'] = self.get_topic_phrases(topic_id)
        return df

    def get_top_bigrams(self, limit = 50):
        sql = "SELECT ngram, ngram_count from ngrambi ORDER BY ngram_count DESC LIMIT {}".format(limit)
        df = pd.read_sql_query(sql, self.corpus.conn)
        df['ngram_percent'] = (df.ngram_count / df.ngram_count.max() * 100).astype('int')
        return df

    def get_topic_list(self, by_alpha = True):
        topics = self.model.get_table('topic', set_index=True)
        alpha_max = topics.topic_alpha.max()

        from scipy import stats
        topics['topic_alpha_zscore'] = stats.zscore(topics.topic_alpha)
        topics['topic_alpha_percent'] = ((topics.topic_alpha / alpha_max) * 100).astype(int)
        topics['topic_alpha_zsign'] = topics.topic_alpha_zscore.apply(lambda x: 'pos' if x > 0 else 'neg')

        num_topics = len(topics.index)
        sql = "SELECT topic_id, GROUP_CONCAT(topic_phrase, ', ') as phrases FROM topicphrase " \
              "GROUP BY topic_id ORDER BY phrase_weight DESC"
        phrases = pd.read_sql_query(sql, self.model.conn)
        phrases.set_index('topic_id', inplace=True)
        cards = []
        if by_alpha:
            topic_id_list = topics.topic_alpha.sort_values(ascending=False).index.tolist()
        else:
            topic_id_list = range(num_topics)
        for topic_id in topic_id_list:
            card = dict(
                topic_id = topic_id,
                topic_alpha = round(topics.loc[topic_id].topic_alpha, 5),
                topic_alpha_zscore = round(topics.loc[topic_id].topic_alpha_zscore, 5),
                topic_alpha_zsign = topics.loc[topic_id].topic_alpha_zsign,
                topic_alpha_percent = topics.loc[topic_id].topic_alpha_percent,
                topic_phrases = phrases.loc[topic_id].phrases,
                topic_words = topics.loc[topic_id].topic_words
            )
            cards.append(card)
        return cards

    def get_topic_phrases(self, topic_id):
        sql = "SELECT topic_phrase FROM topicphrase WHERE topic_id = {} ORDER BY phrase_weight DESC".format(topic_id)
        phrases = ', '.join(pd.read_sql_query(sql, self.model.conn).topic_phrase.tolist())
        return phrases

    def get_topic_entropy_hist(self):
        doctopics = self.model.get_table('doctopic', set_index=True)
        doctopics.unstack()

    # todo: REMOVE
    def get_topic_trend_matrix(self, doc_col = None):
        if not doc_col:
            doc_col = self.config.ini['DEFAULT']['src_ord_col'] # Should wrap these calls with a method
        src_docs = self.corpus.get_table('doc', set_index=True)
        doctopics = pd.read_sql_query('SELECT * FROM doctopic WHERE topic_weight >= {}'.format(self.thresh),
                                      self.model.conn)
        doctopics.set_index(['doc_id', 'topic_id'], inplace=True)
        dtw = doctopics.unstack()
        dtw[doc_col] = src_docs[doc_col]
        dtm = dtw.groupby(doc_col).mean().fillna(0)
        if dtm.columns.nlevels == 2:
            dtm.columns = dtm.columns.droplevel(0)
        return dtm

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
        doc_col = self.config.ini['DEFAULT']['src_ord_col']
        sql = "SELECT {0} as ord_val, `{1}` as topic_weight FROM topicdocord_matrix ORDER BY {0}".format(doc_col, topic_id)
        df = pd.read_sql_query(sql, self.model.conn)
        return df

    # Put this in the database
    def get_max_topic_weight(self):
        sql = 'SELECT MAX(topic_weight) as max_weight FROM doctopic'
        df = pd.read_sql_query(sql, self.model.conn)
        return df.values.tolist()[0][0] # WTF

    # todo: REMOVE
    def get_topic_label_matrix(self, doc_col = None):
        if not doc_col:
            doc_col = self.config.ini['DEFAULT']['src_ord_col'] # Should wrap these calls with a method
        src_docs = self.corpus.get_table('doc', set_index=True)
        #doctopics = trial.modeldb.get_table('doctopic', set_index=True)
        doctopics = pd.read_sql_query('SELECT * FROM doctopic WHERE topic_weight >= {}'.format(self.thresh),
                                      self.model.conn)
        doctopics.set_index(['doc_id', 'topic_id'], inplace=True)
        dtw = doctopics.unstack()
        dtw[doc_col] = src_docs[doc_col]
        dtm = dtw.groupby(doc_col).mean().fillna(0)
        if dtm.columns.nlevels == 2:
            dtm.columns = dtm.columns.droplevel(0)

        # Sort by alpha
        topics = self.model.get_table('topic', set_index=True)
        topics = topics.sort_values('topic_alpha', ascending=False)
        dtm = dtm[topics.index.tolist()]
        dtm.columns = topics.reset_index().apply(lambda x: 'T{} {}'.format(x.topic_id, x.topic_words), axis=1)

        return dtm

    def get_docs_for_topic_and_label(self, topic_id, doc_col_value, doc_col = None):
        if not doc_col:
            doc_col = self.config.ini['DEFAULT']['src_ord_col'] # Should wrap these calls with a method
        src_docs = pd.read_sql_query("SELECT * FROM doc "
                                     "WHERE {} = '{}' LIMIT 100".format(doc_col, doc_col_value), self.corpus.conn)
        return src_docs

    def test(self):
        return self.model.get_table('topic', set_index=True)


