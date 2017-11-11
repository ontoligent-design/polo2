from polo2 import PoloConfig
from polo2 import PoloDb
from gensim import models
import pandas as pd

class PoloGensim:

    def __init__(self, conifg, corpus_dbfile = None):
        self.gs_corpus = None
        self.gs_dict = None
        self.db = PoloDb(corpus_dbfile)

    def make_gs_corpus(self):
        doctokenbow = self.db.get_table('doctokenbow')
        doctokenbow.set_index('doc_id', inplace=True)
        self.gs_corpus = [[(row[0], row[1])
                           for row in doctokenbow.loc[doc_id == doc_id, ['token_id', 'token_count']].values]
                           for doc_id in doctokenbow.index.unique()]
        """
        self.gs_corpus = []
        for doc_id in doctokenbow.index.unique():
            doc = []
            for row in doctokenbow.loc[doc_id, ['token_id', 'token_count']].values:
                doc.append((row[0], row[1]))
            self.gs_corpus.append(doc)
        """

    def make_gs_dict(self):
        token = self.db.get_table('token')
        self.gs_dict = {row[0]: row[1] for row in token[['token_id', 'token_str']].values}

    def get_hdp(self):
        hdp = models.HdpModel(self.gs_corpus, self.gs_dict)
        hdp_topics = hdp.get_topics()
        hdp_df = pd.DataFrame(hdp_topics)
        hdp_dfn = pd.DataFrame(hdp_df.unstack())
        hdp_dfn.reset_index(inplace=True)
        hdp_dfn.columns = ['token_id', 'topic_id', 'token_freq']
        #hdp_dfn.rename(columns = {'level_0': 'topic_id', 'level_1': 'token_id', '0': 'token_freq'}, inplace=True)
        self.db.put_table(hdp_dfn, 'hdp', if_exists='replace')

        # todo: Go the next step and extract topic with word with freqs above a thresh
        thresh = 0.0005
        sql = """
        SELECT topic_id, GROUP_CONCAT(token_str, ' ') AS top_words
        FROM (SELECT topic_id, token_id FROM hdp WHERE token_freq > {} 
        ORDER BY topic_id, token_freq DESC)
        JOIN token USING (token_id) GROUP BY topic_id;
         """.format(thresh)
        hdp_topics = pd.read_sql_query(sql, self.db.conn)
        self.db.put_table(hdp_topics, 'hdd_topics')

        thresh = 0.005 # Note this is different from what's in config.ini
        #hdp_dfn = hpd_dfn[htp_dfn.topic_freq >= thresh]
