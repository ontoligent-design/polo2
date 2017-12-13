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

    def get_prhases_for_topic(self, topic_id):
        sql = "SELECT topic_phrase FROM topicphrase WHERE topic_id = {} ORDER BY phrase_weight DESC".format(topic_id)
        phrases = ', '.join(pd.read_sql_query(sql, self.model.conn).topic_phrase.tolist())
        return phrases

    def get_topic_entropy_hist(self):
        doctopics = self.model.get_table('doctopic', set_index=True)
        doctopics.unstack()

    def get_topic_label_matrix(self, doc_col = None, thresh = 0.05):
        if not doc_col:
            doc_col = self.config.ini['DEFAULT']['src_ord_col'] # Should wrap these calls with a method
        src_docs = self.corpus.get_table('doc', set_index=True)
        #doctopics = trial.modeldb.get_table('doctopic', set_index=True)
        doctopics = pd.read_sql_query('SELECT * FROM doctopic WHERE topic_weight >= {}'.format(thresh),
                                      self.model.conn)
        doctopics.set_index(['doc_id', 'topic_id'], inplace=True)
        dtw = doctopics.unstack()
        dtw[doc_col] = src_docs[doc_col]
        dtm = dtw.groupby(doc_col).mean().fillna(0)
        if dtm.columns.nlevels == 2:
            dtm.columns = dtm.columns.droplevel(0)

        topics = self.model.get_table('topic', set_index=True)
        topics = topics.sort_values('topic_alpha', ascending=False)
        dtm = dtm[topics.index.tolist()]
        dtm.columns = topics.reset_index().apply(lambda x: 'T{} {}'.format(x.topic_id, x.topic_words), axis=1)

        return dtm

    def test(self):
        return self.model.get_table('topic', set_index=True)


