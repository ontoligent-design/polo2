import pandas as pd
from polo2 import PoloDb

class PoloUtil:

    def __init__(self, config):
        self.config = config
        self.corpus = None
        self.model = None
        self.retro = None

    def retro_combine(self, corpus_dbfile, model_dbfile, retro_dbfile=None):
        self.corpus = PoloDb(corpus_dbfile)
        self.model = PoloDb(model_dbfile)
        if retro_dbfile == None:
            retro_dbfile = '{}-retro-combo.db'.format(self.config.ini['DEFAULT']['slug'])
        self.retro = PoloDb(retro_dbfile)
        self.create_retro_db()

    def create_doc_table(self):
        doc = self.model.get_table('doc')
        src_doc = self.corpus.get_table('doc')
        new_doc = pd.DataFrame(columns=['doc_id', 'doc_label', 'doc_str'])
        new_doc['doc_id'] = doc['doc_id']
        doc.set_index('doc_id', inplace=True)
        src_doc.set_index('doc_id', inplace=True)
        new_doc.set_index('doc_id', inplace=True)
        new_doc['doc_label'] = doc.doc_label
        new_doc['doc_str'] = src_doc.doc_content
        self.retro.put_table(new_doc, 'doc', if_exists='replace', index=True)

    def create_src_doc_table(self):
        src_doc = self.corpus.get_table('doc')
        src_doc.set_index('doc_id', inplace=True)
        new_src_doc = pd.DataFrame(columns='src_meta_id doc_id doc_title doc_uri doc_label doc_ord doc_content doc_original doc_year doc_date doc_citation'.split())
        new_src_doc['doc_id'] = src_doc.index
        new_src_doc.set_index('doc_id', inplace=True)
        new_src_doc['doc_title'] = src_doc.doc_title
        new_src_doc['doc_uri'] = src_doc.doc_key
        new_src_doc['doc_uri'] = new_src_doc['doc_uri'].apply(lambda x: self.config.ini['DEFAULT']['src_base_url'] + str(x))
        new_src_doc['doc_label'] = src_doc.doc_label
        new_src_doc['doc_ord'] = None
        new_src_doc['doc_content'] = src_doc.doc_content
        new_src_doc['doc_original'] = src_doc.doc_original
        new_src_doc['doc_year'] = src_doc.doc_year
        new_src_doc['doc_date'] = src_doc.doc_date
        new_src_doc['doc_citation'] = None
        self.retro.put_table(new_src_doc, 'src_doc', if_exists='replace', index=True)

    def create_src_doc_meta_table(self):
        src_doc_meta = pd.DataFrame({'src_meta_id': self.config.ini['DEFAULT']['slug'],
             'src_meta_desc': self.config.ini['DEFAULT']['title'],
             'src_meta_base_url': self.config.ini['DEFAULT']['src_base_url'],
             'src_meta_ord_type': None}, index=['src_meta_id'])
        self.retro.put_table(src_doc_meta, 'src_doc_meta', if_exists='replace')

    def create_word_table(self):
        word = self.corpus.get_table('token')
        new_word = pd.DataFrame(columns='word_id word_str word_freq word_stem'.split())
        new_word['word_id'] = word.index
        new_word.set_index('word_id', inplace=True)
        new_word['word_str'] = word.token_str
        new_word['word_freq'] = word.token_count
        new_word['word_stem'] = None
        self.retro.put_table(new_word, 'word', if_exists='replace', index=True)

    def create_docword_table(self):
        sql = "SELECT dt.doc_id, t.ROWID as 'word_id', t.token_str as 'word_str', t.token_count as 'word_count', NULL as 'tfidf_weight' " \
              "FROM doctoken dt JOIN token t USING(token_str)"
        new_docword = pd.read_sql_query(sql, self.corpus.conn)
        self.retro.put_table(new_docword, 'docword', if_exists='replace')

    def create_config_table(self):
        pass

    def create_doctopic_table(self):
        doctopic = self.model.get_table('doctopic')
        doctopic['topic_label'] = doctopic['topic_id'].apply(lambda x: 't{}'.format(x))
        doctopic = doctopic[['doc_id', 'topic_label', 'topic_weight']]
        doctopic.set_index(['doc_id', 'topic_label'], inplace=True)
        doctopic_wide = doctopic.unstack().reset_index()
        doctopic_wide.columns = doctopic_wide.columns.droplevel(0)
        doctopic_wide.rename(columns={'': 'doc_id'}, inplace=True)
        self.retro.put_table(doctopic_wide, 'doctopic', if_exists='replace')

    def create_topic_table(self):
        topic = self.model.get_table('topic')
        new_topic = pd.DataFrame(columns='topic_id topic_alpha total_tokens topic_words'.split())
        new_topic['topic_id'] = topic.topic_id
        new_topic['topic_alpha'] = topic.topic_alpha
        new_topic['topic_words'] = topic.topic_words
        new_topic['total_tokens'] = topic.topic_tokens
        self.retro.put_table(new_topic, 'topic', if_exists='replace')


    def create_topicphrase_table(self):
        topicphrase = self.model.get_table('topicphrase')
        self.retro.put_table(topicphrase, 'topicphrase', if_exists='replace')

    def create_topicword_table(self):
        topicword = self.model.get_table('topicword')
        word = self.model.get_table('word')
        topicword['word_count'] = topicword['word_count'].astype(int)
        topicword['topic_label'] = topicword['topic_id'].apply(lambda x: 't{}'.format(x))
        topicword = topicword[['word_id', 'topic_label', 'word_count']]
        topicword.set_index(['word_id', 'topic_label'], inplace=True)
        topicword_wide = topicword.unstack().reset_index()
        topicword_wide.columns = topicword_wide.columns.droplevel(0)
        topicword_wide.rename(columns={'': 'word_id'}, inplace=True)
        topicword_wide.fillna(0, inplace=True)
        topicword_wide.set_index('word_id', inplace=True)
        word.set_index('word_id', inplace=True)
        topicword_wide['word_str'] = word.word_str
        self.retro.put_table(topicword_wide, 'topicword', if_exists='replace', index=True)

    def create_doctopic_long_table(self):
        doctopic = self.model.get_table('doctopic')
        self.retro.put_table(doctopic, 'doctopic_long', if_exists='replace')

    def create_topicword_long_table(self):
        topicword = self.model.get_table('topicword')
        word = self.model.get_table('word')
        topicword['word_count'] = topicword['word_count'].astype(int)
        word.set_index('word_id', inplace=True)
        topicword.set_index(['word_id','topic_id'], inplace=True)
        topicword['word_str'] = word.word_str
        self.retro.put_table(topicword, 'topicword_long', if_exists='replace', index=True)

    def create_topicpair_table(self):
        topicpair = self.model.get_table('topicpair')
        new_tp = pd.DataFrame(columns='topic_id1 topic_id2 cosine_sim js_div'.split())
        new_tp['topic_id1'] = topicpair.topic_a
        new_tp['topic_id2'] = topicpair.topic_b
        new_tp['cosine_sim'] = topicpair.cosine_sim
        new_tp['js_div'] = topicpair.js_div
        self.retro.put_table(new_tp, 'topicpair', if_exists='replace')

    def create_topicpair_by_deps_table(self):
        topicpair = self.model.get_table('topicpair')
        new_tp = pd.DataFrame(columns='topic_a topic_b p_a p_b p_ab p_aGb p_bGa i_ab'.split())
        new_tp['topic_a'] = topicpair.topic_a
        new_tp['topic_b'] = topicpair.topic_b
        new_tp['p_a'] = topicpair.p_a
        new_tp['p_b'] = topicpair.p_b
        new_tp['p_ab'] = topicpair.p_ab
        new_tp['p_aGb'] = topicpair.p_aGb
        new_tp['p_bGa'] = topicpair.p_bGa
        new_tp['i_ab'] = topicpair.i_ab
        self.retro.put_table(new_tp, 'topicpair_by_deps')

    def create_doctopic_sig_table(self):
        pass

    # fixme: The sql for tables with topics for columns need to be generated!
    def create_retro_db(self):
        sql_creators = """
        CREATE TABLE IF NOT EXISTS src_doc_meta (src_meta_id TEXT,src_meta_desc TEXT,src_meta_base_url TEXT,src_meta_ord_type TEXT);
        CREATE TABLE IF NOT EXISTS src_doc (src_meta_id TEXT,doc_id INTEGER PRIMARY KEY,doc_title TEXT,doc_uri TEXT UNIQUE,doc_label TEXT,doc_ord INTEGER,doc_content TEXT,doc_original TEXT,doc_year INTEGER,doc_date TEXT,doc_citation TEXT);
        CREATE TABLE IF NOT EXISTS word (word_id INTEGER PRIMARY KEY,word_str TEXT,word_freq INTEGER,word_stem TEXT);
        CREATE TABLE IF NOT EXISTS doc (doc_id INTEGER PRIMARY KEY,doc_label TEXT,doc_str TEXT);
        CREATE TABLE IF NOT EXISTS docword (doc_id INTEGER,word_id INTEGER,word_str TEXT,word_count INTEGER,tfidf_weight REAL);
        CREATE TABLE IF NOT EXISTS config (key TEXT, value TEXT);
        CREATE TABLE IF NOT EXISTS doctopic (doc_id INTEGER PRIMARY KEY, doc_label TEXT, topic_entropy REAL, t0 REAL, t1 REAL, t2 REAL, t3 REAL, t4 REAL, t5 REAL, t6 REAL, t7 REAL, t8 REAL, t9 REAL, t10 REAL, t11 REAL, t12 REAL, t13 REAL, t14 REAL, t15 REAL, t16 REAL, t17 REAL, t18 REAL, t19 REAL, t20 REAL, t21 REAL, t22 REAL, t23 REAL, t24 REAL, t25 REAL, t26 REAL, t27 REAL, t28 REAL, t29 REAL, t30 REAL, t31 REAL, t32 REAL, t33 REAL, t34 REAL, t35 REAL, t36 REAL, t37 REAL, t38 REAL, t39 REAL, t40 REAL, t41 REAL, t42 REAL, t43 REAL, t44 REAL, t45 REAL, t46 REAL, t47 REAL, t48 REAL, t49 REAL, t50 REAL, t51 REAL, t52 REAL, t53 REAL, t54 REAL, t55 REAL, t56 REAL, t57 REAL, t58 REAL, t59 REAL, t60 REAL, t61 REAL, t62 REAL, t63 REAL, t64 REAL, t65 REAL, t66 REAL, t67 REAL, t68 REAL, t69 REAL, t70 REAL, t71 REAL, t72 REAL, t73 REAL, t74 REAL, t75 REAL, t76 REAL, t77 REAL, t78 REAL, t79 REAL);
        CREATE TABLE IF NOT EXISTS topic (topic_id INTEGER PRIMARY KEY, topic_alpha REAL, total_tokens INTEGER, topic_words TEXT);
        CREATE TABLE IF NOT EXISTS topicphrase (topic_id INTEGER, topic_phrase TEXT, phrase_count INTEGER, phrase_weight REAL);
        CREATE TABLE IF NOT EXISTS topicword (word_id INTEGER, word_str TEXT, t0 INTEGER, t1 INTEGER, t2 INTEGER, t3 INTEGER, t4 INTEGER, t5 INTEGER, t6 INTEGER, t7 INTEGER, t8 INTEGER, t9 INTEGER, t10 INTEGER, t11 INTEGER, t12 INTEGER, t13 INTEGER, t14 INTEGER, t15 INTEGER, t16 INTEGER, t17 INTEGER, t18 INTEGER, t19 INTEGER, t20 INTEGER, t21 INTEGER, t22 INTEGER, t23 INTEGER, t24 INTEGER, t25 INTEGER, t26 INTEGER, t27 INTEGER, t28 INTEGER, t29 INTEGER, t30 INTEGER, t31 INTEGER, t32 INTEGER, t33 INTEGER, t34 INTEGER, t35 INTEGER, t36 INTEGER, t37 INTEGER, t38 INTEGER, t39 INTEGER, t40 INTEGER, t41 INTEGER, t42 INTEGER, t43 INTEGER, t44 INTEGER, t45 INTEGER, t46 INTEGER, t47 INTEGER, t48 INTEGER, t49 INTEGER, t50 INTEGER, t51 INTEGER, t52 INTEGER, t53 INTEGER, t54 INTEGER, t55 INTEGER, t56 INTEGER, t57 INTEGER, t58 INTEGER, t59 INTEGER, t60 INTEGER, t61 INTEGER, t62 INTEGER, t63 INTEGER, t64 INTEGER, t65 INTEGER, t66 INTEGER, t67 INTEGER, t68 INTEGER, t69 INTEGER, t70 INTEGER, t71 INTEGER, t72 INTEGER, t73 INTEGER, t74 INTEGER, t75 INTEGER, t76 INTEGER, t77 INTEGER, t78 INTEGER, t79 INTEGER, word_sum INTEGER);
        CREATE TABLE IF NOT EXISTS doctopic_long (doc_id INTEGER NOT NULL, topic_id INTEGER NOT NULL, topic_weight REAL NOT NULL, UNIQUE (doc_id, topic_id));
        CREATE TABLE IF NOT EXISTS topicword_long (word_id INTEGER NOT NULL, word_str TEXT NOT NULL, topic_id INTEGER NOT NULL, word_count INTEGER NOT NULL, UNIQUE (word_id, topic_id));
        CREATE TABLE IF NOT EXISTS topicpair (topic_id1 INTEGER, topic_id2 INTEGER, cosine_sim REAL, js_div REAL);
        CREATE TABLE IF NOT EXISTS topicpair_by_deps (topic_a INTEGER, topic_b INTEGER, p_a REAL, p_b REAL, p_ab REAL, p_aGb REAL, p_bGa REAL, i_ab REAL);
        CREATE TABLE IF NOT EXISTS doctopic_sig (doc_id INTEGER PRIMARY KEY, topic_sig TEXT, topic_sig_sorted TEXT, topic_n INTEGER);
        """.split(';')
        for sql_create in sql_creators:
            self.retro.conn.execute(sql_create)

if __name__ == '__main__':
    pass
