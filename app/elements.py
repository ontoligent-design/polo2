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
        topic_id = int(topic_id)
        sql = 'SELECT * FROM topic WHERE topic_id = ?'
        df = pd.read_sql_query(sql, self.model.conn, params=(topic_id,))
        df.set_index('topic_id', inplace=True)
        df['topic_phrases'] = self.get_topic_phrases(topic_id)
        return df

    def get_topics(self):
        topics = self.model.get_table('topic', set_index=True)
        topics['topic_alpha_zsign'] = topics.topic_alpha_zscore.apply(lambda x: 'pos' if x > 0 else 'neg')
        alpha_max = topics.topic_alpha.max()
        topics['topic_alpha_percent'] = ((topics.topic_alpha / alpha_max) * 100).astype(int)
        topic_phrases = self.model.get_table('topicphrase')
        topics['topic_phrases'] = topic_phrases.groupby('topic_id').apply(lambda x: ', '.join(x.topic_phrase))
        return topics

    def get_top_bigrams(self, limit = 50):
        limit = int(limit)
        sql = "SELECT ngram, ngram_count from ngrambi ORDER BY ngram_count DESC LIMIT {}".format(limit)
        df = pd.read_sql_query(sql, self.corpus.conn)
        df['ngram_percent'] = (df.ngram_count / df.ngram_count.max() * 100).astype('int')
        return df

    def get_topic_phrases(self, topic_id):
        topic_id = int(topic_id)
        sql = "SELECT topic_phrase FROM topicphrase WHERE topic_id = ? ORDER BY phrase_weight DESC"
        phrases = ', '.join(pd.read_sql_query(sql, self.model.conn, params=(topic_id,)).topic_phrase.tolist())
        return phrases

    def get_topic_entropy_hist(self):
        doctopics = self.model.get_table('doctopic', set_index=True)
        doctopics.unstack()

    # fixme: Deprecated function
    def get_topicdoclabel_matrix(self, sort_by_alpha = True):
        dtm = self.model.get_table('topicdoclabel_matrix', set_index=False)
        col1 = dtm.columns.tolist()[0]
        dtm.set_index(col1, inplace=True)

        topics = self.model.get_table('topic', set_index=True)
        if sort_by_alpha:
            topics = topics.sort_values('topic_alpha', ascending=True)
        dtm = dtm[topics.index.astype('str').tolist()]
        if 'topic_gloss' in topics.columns:
            dtm.columns = topics.reset_index().apply(lambda x: 'T{} {}'.format(x.topic_id, x.topic_gloss), axis=1)
        else:
            dtm.columns = topics.reset_index().apply(lambda x: 'T{} {}'.format(x.topic_id, x.topic_words), axis=1)
        return dtm

    # fixme: Remove deprecated function
    """
    def get_topicdoc_matrix(self, sort_by_alpha = True, by='label'):

        if by == 'label':
            dtm = self.model.get_table('topicdoclabel_matrix', set_index=False)
        elif by == 'ord':
            dtm = self.model.get_table('topicdocord_matrix', set_index=False)
        else:
            dtm = self.model.get_table('topicdoclabel_matrix', set_index=False)

        col1 = dtm.columns.tolist()[0]
        dtm.set_index(col1, inplace=True)

        topics = self.model.get_table('topic', set_index=True)
        if sort_by_alpha:
            topics = topics.sort_values('topic_alpha', ascending=True)
        dtm = dtm[topics.index.astype('str').tolist()]
        if 'topic_gloss' in topics.columns:
            dtm.columns = topics.reset_index().apply(lambda x: 'T{} {}'.format(x.topic_id, x.topic_gloss), axis=1)
        else:
            dtm.columns = topics.reset_index().apply(lambda x: 'T{} {}'.format(x.topic_id, x.topic_words), axis=1)
        return dtm
    """

    def get_topicdoc_group_matrix(self, sort_by_alpha = True, group_field='doc_label', use_glass_label=False):
        dtm = self.model.get_table('topic{}_matrix'.format(group_field), set_index=False) # todo: Should be schema driven
        col1 = dtm.columns.tolist()[0] # todo: Should always be doc_group; should be schema driven
        dtm.set_index(col1, inplace=True)

        topics = self.model.get_table('topic', set_index=True)
        if sort_by_alpha:
            topics = topics.sort_values('topic_alpha', ascending=True)
        dtm = dtm[topics.index.astype('str').tolist()]

        if use_glass_label:
            if 'topic_gloss' in topics.columns:
                dtm.columns = topics.reset_index().apply(lambda x: 'T{} {}'.format(x.topic_id, x.topic_gloss), axis=1)
            else:
                dtm.columns = topics.reset_index().apply(lambda x: 'T{} {}'.format(x.topic_id, x.topic_words), axis=1)
        return dtm

    # fixme: Deprecated function
    """
    def get_topicdocord_matrix(self):
        dtm = self.model.get_table('topicdocord_matrix', set_index=False)
        col1 = dtm.columns.tolist()[0]
        dtm.set_index(col1, inplace=True)
        return dtm
    """

    def get_topicdocgrooup_counts(self, table_name):
        doc_counts = pd.DataFrame(self.model.get_table(table_name))
        doc_counts.set_index('doc_group', inplace=True)
        return doc_counts

    def get_topicdoc_sum_matrix(self, dtm, group_counts):
        df = dtm.apply(lambda x: x * group_counts.doc_count.values, axis=0)
        return df

    def get_topicdoc_ord_for_topic(self, topic_id):
        topic_id = int(topic_id)
        doc_col = self.config.ini['DEFAULT']['src_ord_col']
        src_ord_col = self.config.ini['DEFAULT']['src_ord_col']
        table_name = 'topic{}_matrix'.format(src_ord_col)
        sql = "SELECT  doc_group, `{1}` as topic_weight FROM {0} ORDER BY doc_group".format(table_name, topic_id)
        df = pd.read_sql_query(sql, self.model.conn)
        return df

    def get_docs_for_topic(self, topic_id, limit=10):
        sql = "SELECT src_doc_id, topic_weight, topic_weight_zscore FROM doctopic " \
              "JOIN doc USING(doc_id) WHERE topic_id = ? " \
              "ORDER BY topic_weight DESC LIMIT {}".format(limit)
        df = pd.read_sql_query(sql, self.model.conn, params=(topic_id,))
        df.set_index('src_doc_id', inplace=True)
        doc_ids = ','.join(df.index.astype('str').tolist())
        sql2 = "SELECT doc_id, doc_title, doc_content, doc_key FROM doc WHERE doc_id IN ({})".format(doc_ids)
        df2 = pd.read_sql_query(sql2, self.corpus.conn,)
        df2.set_index('doc_id', inplace=True)
        df = df.join(df2)
        return df

    def get_docs_for_topic_and_label(self, topic_id, doc_col_value, doc_col = None, limit = 100):
        if not doc_col:
            doc_col = self.config.ini['DEFAULT']['src_ord_col'] # Should wrap these calls with a method
        df = pd.read_sql_query("SELECT doc_id, doc_title, doc_content, doc_key  FROM doc WHERE {} = ? LIMIT {}".format(doc_col, limit),
                               self.corpus.conn, params=(doc_col_value,))
        df.set_index('doc_id', inplace=True)
        doc_ids = ','.join(df.index.astype('str').tolist())
        sql2 = "SELECT d.*, topic_weight FROM doc d " \
               "JOIN doctopic dt USING(doc_id) " \
               "WHERE src_doc_id IN ({}) AND topic_id = ? " \
               "ORDER BY topic_weight DESC LIMIT 10 ".format(doc_ids)
        df2 = pd.read_sql_query(sql2, self.model.conn, params=(topic_id,))
        df2.set_index('src_doc_id', inplace=True)
        df = df.join(df2)
        return df.sort_values('topic_weight', ascending=False)

    def get_docs_for_topic_entropy(self, topic_entropy, limit = 100):
        topic_entropy_min = float(topic_entropy) - .05
        topic_entropy_max = float(topic_entropy) + .05
        sql = "SELECT src_doc_id, topic_entropy, topic_entropy_zscore FROM doc " \
              "WHERE topic_entropy >= ? AND topic_entropy < ? " \
              "ORDER BY src_doc_id LIMIT {} ".format(limit)
        df = pd.read_sql_query(sql, self.model.conn, params=(topic_entropy_min, topic_entropy_max))
        df.set_index('src_doc_id', inplace=True)
        doc_ids = ','.join(df.index.astype('str').tolist())
        sql2 = "SELECT doc_id, doc_title, doc_content, doc_key, doc_label " \
               "FROM doc WHERE doc_id IN ({})".format(doc_ids)
        df2 = pd.read_sql_query(sql2, self.corpus.conn,)
        df2.set_index('doc_id', inplace=True)
        df = df.join(df2)
        return df

    # todo: Put this in database?
    def get_doc_entropy(self):
        sql = "SELECT ROUND(topic_entropy, 1) as h, count() as n FROM doc GROUP BY h ORDER BY h"
        df = pd.read_sql_query(sql, self.model.conn)
        return df

    # todo: Put this in database
    def get_doc_entropy_avg(self):
        sql = "SELECT ROUND(AVG(topic_entropy), 1) as h_avg FROM doc"
        df = pd.read_sql_query(sql, self.model.conn)
        return df['h_avg'].tolist()[0]

    def test(self):
        return 1

    def get_topicpair_matrix(self, sim=None, symmetric=True):
        """Get topic pair matrix by similarity or contiguity measure.
         sim values include cosim, jscore, and i_ab"""
        pairs = self.model.get_table('topicpair', set_index=True)
        if symmetric:
            tpm = pairs.append(pairs.reorder_levels(['topic_b_id', 'topic_a_id'])).unstack()
        else:
            tpm = pairs.unstack()

        if sim:
            return tpm[sim]
        else:
            return tpm

    def get_topicpair_net(self, thresh = 0.05):
        topics = self.model.get_table('topic')
        pairs = self.model.get_table('topicpair', set_index=False)
        pairs = pairs.loc[pairs.i_ab >= thresh, ['topic_a_id', 'topic_b_id', 'i_ab']]
        nodes = [{'id':t, 'label':'T{}: {} '.format(t, topics.loc[t].topic_gloss)} for t in pd.concat([pairs.topic_a_id, pairs.topic_b_id], axis=0).unique()]
        edges = [{'from': int(pairs.loc[i].topic_a_id), 'to': int(pairs.loc[i].topic_b_id)} for i in pairs.index]
        return nodes, edges

    def get_topics_related(self, topic_id):
        sql1 = "SELECT topic_b_id as topic_id, jsd, jscore, p_ab, p_aGb, p_bGa, i_ab " \
               "FROM topicpair WHERE topic_a_id = ?".format(topic_id)
        sql2 = "SELECT topic_a_id as topic_id, jsd, jscore, p_ab, p_aGb, p_bGa, i_ab " \
               "FROM topicpair WHERE topic_b_id = ?".format(topic_id)
        df1 = pd.read_sql_query(sql1, self.model.conn, params=(topic_id,))
        df2 = pd.read_sql_query(sql2, self.model.conn, params=(topic_id,))
        df1 = df2.append(df1)
        df1.sort_values('topic_id', inplace=True)
        df1.set_index('topic_id', inplace=True)
        return df1

    def get_group_matrix(self, group_field):
        df = self.model.get_table('topic{}_matrix'.format(group_field))
        return df.set_index('doc_group')

    def get_group_pairs(self, group_field):
        return self.model.get_table('topic{}_pairs'.format(group_field))

    def get_group_counts(self, group_field):
        df = self.model.get_table('topic{}_matrix_counts'.format(group_field))
        return df.set_index('doc_group')

    def get_group_topics(self, group_field, group_name):
        table_name = 'topic{}_matrix'.format(group_field)
        sql = 'SELECT * FROM {} WHERE doc_group = ?'.format(table_name)
        df = pd.read_sql_query(sql, self.model.conn, params=(group_name,))
        df.set_index('doc_group', inplace=True)
        df = df.T
        df.index.name = 'topic_id'
        df.columns = ['topic_weight']
        topics = self.model.get_table('topic', set_index=True)
        df['topic_gloss'] = topics.topic_gloss.tolist()
        df['label'] = 'T' + df.index + ' ' + df.topic_gloss
        return df

    def get_group_comps(self, group_field, group_name):
        table_name = 'topic{}_pairs'.format(group_field)
        sql1 = "SELECT group_b as 'doc_group', kld, jsd, jscore, euclidean FROM {} WHERE group_a = ?".format(table_name)
        sql2 = "SELECT group_a as 'doc_group', kld, jsd, jscore, euclidean FROM {} WHERE group_b = ?".format(table_name)
        df1 = pd.read_sql_query(sql1, self.model.conn, params=(group_name,))
        df2 = pd.read_sql_query(sql2, self.model.conn, params=(group_name,))
        return df1.append(df2).sort_values('doc_group').set_index('doc_group')

    def get_group_docs(self, group_field, group_name):
        """Get a random selection of documents associated with the group name"""
        # Get a random list of src docs for the group_name
        # Get doc info for each
        pass

    def get_max_topic_weight(self):
        sql = "SELECT value as 'max_tw' FROM config WHERE key = 'doctopic_weight_max'"
        df = pd.read_sql_query(sql, self.model.conn)
        return df.max_tw.tolist()[0]