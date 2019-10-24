import os
import time
import re
import pandas as pd
from itertools import combinations
from lxml import etree
from scipy import stats
from polo2 import PoloDb
from polo2 import PoloFile
from polo2 import PoloMath as pm


class PoloMallet(PoloDb):

    def __init__(self, config, trial='trial1'):
        """Initialize MALLET with trial name"""
        if trial not in config.trials:
            raise ValueError("Invalid trail name `{}`.format(trial)")

        self.config = config
        self.trial = trial
        self.config.set_config_attributes(self)
        self.config.set_config_attributes(self, self.trial)

        # todo: Put this in config.ini
        self.cfg_tw_quantile = 0.8

        # Temporary hack to handle casting
        for key in "num_topics num_iterations optimize_interval num_threads num_top_words".split():
            att = 'cfg_{}'.format(key)
            setattr(self, att, int(getattr(self, att)))
        self.cfg_thresh = float(self.cfg_thresh)

        self.trial_name = self.trial  # HACK
        self.file_prefix = '{}/{}'.format(self.cfg_mallet_out_dir, self.trial_name)
        self.mallet = {'import-file': {}, 'train-topics': {}}
        self.mallet_init()

        dbfile = self.config.generate_model_db_file_path(self.trial)
        PoloDb.__init__(self, dbfile)

    # todo: Remove or replace
    def generate_trial_name(self):
        """Generate trial name based on metadata"""
        ts = time.time()
        self.trial_name = '{}-model-t{}-i{}-{}'.format(self.trial, self.cfg_num_topics,
                                                              self.cfg_num_iterations, int(ts))
    def mallet_init(self):
        """Initialize command line arguments for MALLET"""
        # todo: Consider putting trunhis in the init for the object itself
        if not os.path.exists(self.cfg_mallet_path):
            raise ValueError('Mallet cannot be found.')

        self.mallet['import-file']['input'] = self.cfg_mallet_corpus_input
        self.mallet['import-file']['output'] = '{}/mallet-corpus.mallet'.format(self.cfg_mallet_out_dir) # Put this in corpus?
        self.mallet['import-file']['keep-sequence'] = 'TRUE' # todo: Control this by config
        self.mallet['import-file']['remove-stopwords'] = 'FALSE' # todo: Control this by config

        self.mallet['train-topics']['num-topics'] = self.cfg_num_topics
        self.mallet['train-topics']['num-top-words'] = self.cfg_num_top_words
        self.mallet['train-topics']['num-iterations'] = self.cfg_num_iterations
        self.mallet['train-topics']['optimize-interval'] = self.cfg_optimize_interval
        self.mallet['train-topics']['num-threads'] = self.cfg_num_threads
        self.mallet['train-topics']['input'] = self.mallet['import-file']['output']
        self.mallet['train-topics']['output-topic-keys'] = '{}-topic-keys.txt'.format(self.file_prefix)
        self.mallet['train-topics']['output-doc-topics'] = '{}-doc-topics.txt'.format(self.file_prefix)
        self.mallet['train-topics']['word-topic-counts-file'] = '{}-word-topic-counts.txt'.format(self.file_prefix)
        self.mallet['train-topics']['topic-word-weights-file'] = '{}-topic-word-weights.txt'.format(self.file_prefix)
        self.mallet['train-topics']['xml-topic-report'] = '{}-topic-report.xml'.format(self.file_prefix)
        self.mallet['train-topics']['xml-topic-phrase-report'] = '{}-topic-phrase-report.xml'.format(self.file_prefix)
        self.mallet['train-topics']['diagnostics-file'] = '{}-diagnostics.xml'.format(self.file_prefix)
        # self.mallet['train-topics']['output-topic-docs'] = '{}-topic-docs.txt'.format(self.file_prefix)
        # self.mallet['train-topics']['doc-topics-threshold'] = self.config.thresh
        self.mallet['train-topics']['output-state'] = '{}-state.gz'.format(self.file_prefix)
        self.mallet['train-topics']['num-top-docs'] = self.cfg_num_topics
        self.mallet['train-topics']['doc-topics-max'] = self.cfg_doc_topics_max
        self.mallet['train-topics']['show-topics-interval'] = self.cfg_show_topics_interval

    def mallet_run_command(self, op):
        """Run a MALLET command (e.g. import-file or train-topics)"""
        my_args = ['--{} {}'.format(arg,self.mallet[op][arg]) for arg in self.mallet[op]]
        my_cmd = '{} {} {}'.format(self.cfg_mallet_path, op, ' '.join(my_args))
        print(my_cmd)
        try:
            os.system(my_cmd)
        except:
            raise ValueError('Command would not execute:', my_cmd)

    def mallet_import(self):
        """Import contents of MALLET output files into Polo DB"""
        self.mallet_run_command('import-file')

    def mallet_train(self):
        """Train MALLET by running train-topics"""
        self.mallet_run_command('train-topics')

    def clean_up(self):
        """Clean up files created by MALLET"""
        file_mask = '{}-*.*'.format(self.file_prefix)
        my_cmd = 'rm {}'.format(file_mask)
        try:
            os.system(my_cmd)
        except:
            raise ValueError('Unable to delete files: {}'.format(file_mask))

    # TABLE IMPORT METHODS

    def tables_to_db(self):
        """Import core tables from MALLET files into Polo DB"""
        self.import_table_config()
        self.import_table_state()
        self.import_table_topic()
        self.import_tables_topicword_and_word()
        self.import_table_doctopic()
        self.import_table_topicphrase()

    def import_table_state(self, src_file=None):
        """Import the state file into docword table"""
        if not src_file:
            src_file = self.mallet['train-topics']['output-state']
        import gzip
        with gzip.open(src_file, 'rb') as f:
            docword = pd.DataFrame([line.split() for line in f.readlines()[3:]],
                                   columns=['doc_id', 'src', 'word_pos', 'word_id', 'word_str', 'topic_id'])
            docword = docword[['doc_id', 'word_id', 'word_pos', 'topic_id']]
            docword = docword.astype('int')
            docword.set_index(['doc_id', 'word_id'], inplace=True)
            self.put_table(docword, 'docword', index=True)

    def import_table_topic(self, src_file=None):
        """Import data into topic table"""
        if not src_file: src_file = self.mallet['train-topics']['output-topic-keys']
        topic = pd.read_csv(src_file, sep='\t', header=None, index_col=False,
                            names=['topic_id', 'topic_alpha', 'topic_words'])
        topic.set_index('topic_id', inplace=True)
        topic['topic_alpha_zscore'] = stats.zscore(topic.topic_alpha)
        topic['topic_gloss'] = 'TBA'
        self.put_table(topic, 'topic', index=True)

    def import_tables_topicword_and_word(self, src_file=None):
        """Import data into topicword and word tables"""
        if not src_file: src_file = self.mallet['train-topics']['word-topic-counts-file']
        WORD = []
        TOPICWORD = []
        src = PoloFile(src_file)
        for line in src.read_lines():
            row = line.strip().split()
            (word_id, word_str) = row[0:2]
            WORD.append((int(word_id), word_str))
            for item in row[2:]:
                (topic_id, word_count) = item.split(':')
                TOPICWORD.append((int(word_id), int(topic_id), int(word_count)))
        word = pd.DataFrame(WORD, columns=['word_id', 'word_str'])
        topicword = pd.DataFrame(TOPICWORD, columns=['word_id', 'topic_id', 'word_count'])
        word.set_index('word_id', inplace=True)
        topicword.set_index(['word_id', 'topic_id'], inplace=True)
        self.put_table(word, 'word', index=True)
        self.put_table(topicword, 'topicword', index=True)

    def import_table_doctopic(self, src_file=None):
        """Import data into doctopic table"""
        if not src_file: src_file = self.mallet['train-topics']['output-doc-topics']
        if 'doc-topics-threshold' in self.mallet['train-topics']:
            DOC = []
            DOCTOPIC = []
            src = PoloFile(src_file)
            for line in src[1:]:
                row = line.split('\t')
                row.pop()  # Pretty sure this is right
                doc_id = row[0]
                src_doc_id = int(row[1].split(',')[0])
                doc_label = row[1].split(',')[1]
                DOC.append([doc_id, src_doc_id, doc_label])
                for i in range(2, len(row), 2):
                    topic_id = row[i]
                    topic_weight = row[i + 1]
                    DOCTOPIC.append([doc_id, topic_id, topic_weight])
            doctopic = pd.DataFrame(DOCTOPIC, columns=['doc_id', 'topic_id', 'topic_weight'])
            doctopic.set_index(['doc_id', 'topic_id'], inplace=True)
            doctopic['topic_weight_zscore'] = stats.zscore(doctopic.topic_weight)
            self.computed_thresh = round(doctopic.topic_weight.quantile(self.cfg_tw_quantile), 3)
            doc = pd.DataFrame(DOC, columns=['doc_id', 'src_doc_id', 'doc_label'])
            doc.set_index('doc_id', inplace=True)
            self.put_table(doctopic, 'doctopic', index=True)
            self.put_table(doc, 'doc', index=True)
        else:
            doctopic = pd.read_csv(src_file, sep='\t', header=None)
            doc = pd.DataFrame(doctopic.iloc[:, 1])
            doc.columns = ['doc_tmp']
            doc['src_doc_id'] = doc.doc_tmp.apply(lambda x: int(x.split(',')[0]))
            doc['doc_label'] = doc.doc_tmp.apply(lambda x: x.split(',')[1])
            doc = doc[['src_doc_id', 'doc_label']]
            doc.index.name = 'doc_id'
            self.put_table(doc, 'doc', index=True)
            doctopic.drop(1, axis = 1, inplace=True)
            doctopic.rename(columns={0:'doc_id'}, inplace=True)
            y = [col for col in doctopic.columns[1:]]
            doctopic_narrow = pd.lreshape(doctopic, {'topic_weight': y})
            doctopic_narrow['topic_id'] = [i for i in range(self.cfg_num_topics)
                                           for doc_id in doctopic['doc_id']]
            doctopic_narrow = doctopic_narrow[['doc_id', 'topic_id', 'topic_weight']]
            doctopic_narrow.set_index(['doc_id', 'topic_id'], inplace=True)
            doctopic_narrow['topic_weight_zscore'] = stats.zscore(doctopic_narrow.topic_weight)
            self.computed_thresh = round(doctopic_narrow.topic_weight\
                                         .quantile(self.cfg_tw_quantile), 3)
            self.put_table(doctopic_narrow, 'doctopic', index=True)

        # todo: Revisit this; in the best place to do this?
        self.set_config_item('computed_thresh', self.computed_thresh)

    def import_table_topicphrase(self, src_file=None):
        """Import data into topicphrase table"""
        if not src_file: src_file = self.mallet['train-topics']['xml-topic-phrase-report']
        TOPICPHRASE = []
        src = PoloFile(src_file)
        tree = etree.parse(src.file)
        for topic in tree.xpath('/topics/topic'):
            topic_id = int(topic.xpath('@id')[0])
            for phrase in topic.xpath('phrase'):
                phrase_weight = float(phrase.xpath('@weight')[0])
                phrase_count = int(phrase.xpath('@count')[0])
                topic_phrase = phrase.xpath('text()')[0]
                TOPICPHRASE.append((topic_id, topic_phrase, phrase_weight, phrase_count))
        topicphrase = pd.DataFrame(TOPICPHRASE, columns=['topic_id', 'topic_phrase',
                                                         'phrase_weight', 'phrase_count'])
        topicphrase.set_index(['topic_id', 'topic_phrase'], inplace=True)
        self.put_table(topicphrase, 'topicphrase', index=True)

    def add_topic_glosses(self):
        """Add glosses to topic table"""
        sql = """
        SELECT topic_id, topic_phrase as topic_gloss, 
            MAX(phrase_weight) as max_phrase_weight
        FROM topicphrase
        GROUP BY topic_id
        """
        topicphrase = pd.read_sql_query(sql, self.conn)
        topicphrase.set_index('topic_id', inplace=True)
        topic = self.get_table('topic', set_index=True)
        topic['topic_gloss'] = topicphrase.topic_gloss
        self.put_table(topic, 'topic', index=True)

    def import_table_config(self):
        """Import data into config table"""
        # fixme: Make this automatic; find a way to dump all values
        cfg = {}
        cfg['trial'] = self.trial
        cfg['dbfile'] = self.dbfile
        cfg['thresh'] = self.cfg_thresh
        cfg['slug'] = self.cfg_slug
        cfg['num_topics'] = self.cfg_num_topics
        cfg['base_path'] = self.cfg_base_path
        cfg['file_prefix'] = self.file_prefix
        config = pd.DataFrame({'key': list(cfg.keys()), 'value': list(cfg.values())})
        self.put_table(config, 'config')

    def add_diagnostics(self, src_file=None):
        """Add diagnostics data to topics and topicword_diags tables"""
        if not src_file: src_file = self.mallet['train-topics']['diagnostics-file']
        TOPIC = []
        TOPICWORD = []
        tkeys = ['id', 'tokens', 'document_entropy', 'word-length', 'coherence',
                 'uniform_dist', 'corpus_dist',
                 'eff_num_words', 'token-doc-diff', 'rank_1_docs',
                 'allocation_ratio', 'allocation_count',
                 'exclusivity']
        tints = ['id', 'tokens']
        wkeys = ['rank', 'count', 'prob', 'cumulative', 'docs', 'word-length', 'coherence',
                 'uniform_dist', 'corpus_dist', 'token-doc-diff', 'exclusivity']
        wints = ['rank', 'count', 'docs', 'word-length']

        src = PoloFile(src_file)
        tree = etree.parse(src.file)
        for topic in tree.xpath('/model/topic'):
            tvals = []
            for key in tkeys:
                xpath = '@{}'.format(key)
                if key in tints:
                    tvals.append(int(float(topic.xpath(xpath)[0])))
                else:
                    tvals.append(float(topic.xpath(xpath)[0]))
            TOPIC.append(tvals)
            for word in topic.xpath('word'):
                wvals = []
                topic_id = tvals[0]  # Hopefully
                wvals.append(topic_id)
                word_str = word.xpath('text()')[0]
                wvals.append(word_str)
                for key in wkeys:
                    xpath = '@{}'.format(key)
                    if key in wints:
                        wvals.append(int(float(word.xpath(xpath)[0])))
                    else:
                        wvals.append(float(word.xpath(xpath)[0]))
                TOPICWORD.append(wvals)
        tkeys = ['topic_{}'.format(re.sub('-', '_', k)) for k in tkeys]
        wkeys = ['topic_id', 'word_str'] + wkeys
        wkeys = [re.sub('-', '_', k) for k in wkeys]

        topic_diags = pd.DataFrame(TOPIC, columns=tkeys)
        topic_diags.set_index('topic_id', inplace=True)
        topics = self.get_table('topic', set_index=True)
        topics = pd.concat([topics, topic_diags], axis=1)
        self.put_table(topics, 'topic', index=True)

        topicword_diags = pd.DataFrame(TOPICWORD, columns=wkeys)
        topicword_diags.set_index(['topic_id', 'word_str'], inplace=True)
        word = self.get_table('word')
        word.set_index('word_str', inplace=True)
        topicword_diags = topicword_diags.join(word, how='inner')
        topicword_diags.reset_index(inplace=True)
        topicword_diags.set_index(['topic_id', 'word_id'], inplace=True)
        self.put_table(topicword_diags, 'topicword_diag', index=True)

    # fixme: Deleting mallet files seems not to be working
    def del_mallet_files(self):
        """Delete MALLET files"""
        file_keys = ['output-topic-keys', 'output-doc-topics',
                     'word-topic-counts-file', 'xml-topic-report',
                     'xml-topic-phrase-report', 'diagnostics-file',
                     'topic-word-weights-file']
        for fk in file_keys:
            if os.path.isfile(self.mallet['train-topics'][fk]):
                print("Deleting {}".format(fk))
                os.remove(str(self.mallet['train-topics'][fk]))

    # UPDATE OR ADD TABLES WITH STATS

    # todo: Consider moving into method that creates doc and doctopic tables
    def add_topic_entropy(self):
        """Add entropy to topic table"""
        doctopic = self.get_table('doctopic')
        doc = self.get_table('doc')
        #topic_entropy = doctopic.groupby('doc_id')['topic_weight'].apply(lambda x: pm.entropy(x))
        #doc['topic_entropy'] = topic_entropy
        doc['topic_entropy'] = doctopic.groupby('doc_id')['topic_weight'].apply(pm.entropy)
        doc['topic_entropy_zscore'] = stats.zscore(doc.topic_entropy)
        doc.set_index('doc_id', inplace=True)
        self.put_table(doc, 'doc', index=True)

    def create_table_topicpair(self):
        """Create topicpair table"""
        thresh = self.get_thresh()

        # Get doc count to calculate topic frequencies
        r = self.conn.execute("select count() from doc")
        doc_num = int(r.fetchone()[0])

        # Create the doctopic matrix dataframe
        # todo: Find out if this can pull from an existing table
        doctopic = self.get_table('doctopic', set_index=True)
        dtm = doctopic['topic_weight'].unstack()
        if dtm.columns.nlevels == 2:
            dtm.columns = dtm.columns.droplevel()
        del doctopic

        # Add topic frequency data to topic table
        topic = self.get_table('topic', set_index=True)
        topic['topic_freq'] = topic.apply(lambda x: len(dtm[dtm[x.name] >= thresh]), axis=1)
        topic['topic_rel_freq'] = topic.apply(lambda x: x.topic_freq / doc_num, axis=1)
        self.put_table(topic, 'topic', index=True)

        # Create topicword matrix dataframe
        topicword = self.get_table('topicword', set_index=True)
        topicword['word_count'] = topicword['word_count'].astype(int)
        twm = topicword.unstack().fillna(0)
        if twm.columns.nlevels == 2:
            twm.columns = twm.columns.droplevel(0)
        del topicword

        # Create topicpair dataframe
        #from itertools import combinations
        pairs = [pair for pair in combinations(topic.index, 2)]
        topicpair = pd.DataFrame(pairs, columns=['topic_a_id', 'topic_b_id'])

        # Calculate distances by word vector
        topicpair['cosim'] = topicpair.apply(lambda x: pm.cosine_sim(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        topicpair['jscore'] = topicpair.apply(lambda x: pm.jscore(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        topicpair['jsd'] = topicpair.apply(lambda x: pm.js_divergence(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)

        # Keep these -- remove the above from the app
        topicpair['cosine_dist'] = topicpair.apply(lambda x: pm.cosine_dist(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        topicpair['js_dist'] = topicpair.apply(lambda x: pm.js_dist(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        topicpair['jaccard_dist'] = topicpair.apply(lambda x: pm.jaccard_dist(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        topicpair['euclidean'] = topicpair.apply(lambda x: pm.euclidean_dist(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        # topicpair['seuclidean'] = topicpair.apply(lambda x: pm.standard_euclidean_dist(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        topicpair['chebyshev'] = topicpair.apply(lambda x: pm.chebyshev_dist(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        topicpair['manhattan'] = topicpair.apply(lambda x: pm.manhattan_dist(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)

        # Calculate PWMI
        def get_p_ab(a, b):
            p_ab = len(dtm[(dtm[a] >= thresh) & (dtm[b] >= thresh)]) / doc_num
            return p_ab
        topicpair['p_ab'] = topicpair.apply(lambda x: get_p_ab(x.topic_a_id, x.topic_b_id), axis=1)
        topicpair['p_aGb'] = topicpair.apply(lambda x: x.p_ab / topic.loc[x.topic_b_id, 'topic_rel_freq'], axis=1)
        topicpair['p_bGa'] = topicpair.apply(lambda x: x.p_ab / topic.loc[x.topic_a_id, 'topic_rel_freq'], axis=1)
        def get_pwmi(a, b, p_ab):
            p_a = topic.loc[a, 'topic_rel_freq']
            p_b = topic.loc[b, 'topic_rel_freq']
            i_ab = pm.pwmi(p_a, p_b, p_ab)
            return i_ab
        topicpair['i_ab'] = topicpair.apply(lambda x: get_pwmi(x.topic_a_id, x.topic_b_id, x.p_ab), axis=1)
        topicpair['x_ab'] = topicpair.apply(lambda x: (x.p_aGb + x.p_bGa) / 2, axis=1)

        # Gravity
        topicpair['gravity'] = topicpair.p_ab / topicpair.js_dist**2

        topicpair.set_index(['topic_a_id', 'topic_b_id'], inplace=True)
        self.put_table(topicpair, 'topicpair', index=True)

    # fixme: Remove deprecated function
    def create_topicdoc_col_matrix(self, group_col):
        """Create topicdoc matrix table for a group column"""

        # Get source doc table
        corpus_db_file = self.config.generate_corpus_db_file_path()
        corpus = PoloDb(corpus_db_file)
        src_docs = corpus.get_table('doc')
        src_docs.rename(columns={'doc_id':'src_doc_id'}, inplace=True)
        del corpus

        # Add the model doc_id to src_doc
        docs = self.get_table('doc')
        src_docs = src_docs.merge(docs[['doc_id', 'src_doc_id']], on='src_doc_id', how='right')
        src_docs.set_index('doc_id', inplace=True) # Change index to align with doctopics
        del docs

        # Get doctopic table
        # thresh = self.get_thresh()
        # doctopics = pd.read_sql_query('SELECT * \
        # FROM doctopic WHERE topic_weight >= ?', self.conn, params=(thresh,))
        # doctopics.set_index(['doc_id', 'topic_id'], inplace=True)
        doctopics = self.get_table('doctopic', set_index=True)
        dtw = doctopics['topic_weight'].unstack()
        del doctopics

        # todo: Streamline the logic here
        if group_col == 'ord':
            doc_col = self.config.ini['DEFAULT']['src_ord_col']
        elif group_col == 'label':
            doc_col = 'doc_label'
        else:
            group_col = 'ord'
            doc_col = self.config.ini['DEFAULT']['src_ord_col']

        dtw['doc_group'] = src_docs[doc_col]
        dtg = dtw.groupby('doc_group')
        dtm = dtg.mean().fillna(0)
        if dtm.columns.nlevels == 2:
            dtm.columns = dtm.columns.droplevel(0)
        self.put_table(dtm, 'topicdoc{}_matrix'.format(group_col), index=True)
        dtm_counts = dtg[0].count().fillna(0)
        dtm_counts.name = 'doc_count'
        self.put_table(dtm_counts, 'topicdoc{}_matrix_counts'.format(group_col), index=True)

    def create_topicdoc_group_matrix(self, group_field='doc_label'):
        """Create topicdoc group matrix table"""

        # Get source doc table
        corpus_db_file = self.config.generate_corpus_db_file_path()
        corpus = PoloDb(corpus_db_file)
        src_docs = corpus.get_table('doc')
        if group_field not in src_docs.columns:
            raise ValueError('Column `{}` does not exist on corpus doc table.'.format(group_field))
        src_docs.rename(columns={'doc_id':'src_doc_id'}, inplace=True)
        del corpus

        # Add the model doc_id to src_doc
        docs = self.get_table('doc')
        src_docs = src_docs.merge(docs[['doc_id', 'src_doc_id']], on='src_doc_id', how='right')
        src_docs.set_index('doc_id', inplace=True) # Change index to align with doctopics
        del docs

        # Get doctopic table
        doctopics = self.get_table('doctopic', set_index=True)
        dtw = doctopics['topic_weight'].unstack()
        del doctopics

        dtw['doc_group'] = src_docs[group_field]
        dtg = dtw.groupby('doc_group')
        dtm = dtg.mean().fillna(0)
        if dtm.columns.nlevels == 2:
            dtm.columns = dtm.columns.droplevel(0)
        self.put_table(dtm, 'topic{}_matrix'.format(group_field), index=True)
        dtm_counts = dtg[0].count().fillna(0)
        dtm_counts.name = 'doc_count'
        self.put_table(dtm_counts, 'topic{}_matrix_counts'.format(group_field), index=True)

    def create_topicdoc_group_pairs(self, group_field='doc_label'):
        """Create topicdoc group pairs table"""
        thresh = self.get_thresh()
        gtm = self.get_table('topic{}_matrix'.format(group_field))
        gtm.set_index('doc_group', inplace=True)
        pairs = [pair for pair in combinations(gtm.index, 2)]
        pair = pd.DataFrame(pairs, columns=['group_a', 'group_b'])
        pair['cosim'] = pair.apply(lambda x: pm.cosine_sim(gtm.loc[x.group_a], gtm.loc[x.group_b]), axis=1)
        pair['jsd'] = pair.apply(lambda x: pm.js_divergence(gtm.loc[x.group_a], gtm.loc[x.group_b]), axis=1)
        pair['jscore'] = pair.apply(lambda x:
                                    pm.jscore(gtm.loc[x.group_a], gtm.loc[x.group_b], thresh=thresh), axis=1)
        pair['euclidean'] = pair.apply(lambda x: pm.euclidean(gtm.loc[x.group_a], gtm.loc[x.group_b]), axis=1)
        pair['kld'] = pair.apply(lambda x: pm.kl_distance(gtm.loc[x.group_a], gtm.loc[x.group_b]), axis=1)
        self.put_table(pair, 'topic{}_pairs'.format(group_field))

    def add_group_field_tables(self):
        """Create topicdoc group matrix tables for group fields in INI"""
        for group_field in self.config.get_group_fields():
            self.create_topicdoc_group_matrix(group_field)
            self.create_topicdoc_group_pairs(group_field)

    def get_thresh(self):
        """Compute the topic weight threshold"""
        config = self.get_table('config')
        if len(config[config.key == 'computed_thresh'].values):
            thresh = config[config.key == 'computed_thresh']['value'].astype('float').tolist()[0]
        else:
            thresh = self.cfg_thresh
        return thresh

    def add_topic_alpha_stats(self):
        """Add topic alpha stats to config table"""
        topic = self.get_table('topic')
        items = dict(
            topic_alpha_max=topic.topic_alpha.max(),
            topic_alpha_min=topic.topic_alpha.min(),
            topic_alpha_avg=topic.topic_alpha.mean()
        )
        self.set_config_items(items)

    def add_maxtopic_to_word(self):
        """Add idxmax topic for each word"""
        topicword = self.get_table('topicword')
        word = self.get_table('word')
        # word['maxtopic'] = topicword.set_index(['topic_id','word_id']).word_count\
        #     .unstack().fillna(0).idxmax()
        twm = topicword.set_index(['word_id', 'topic_id']).word_count.unstack().fillna(0)
        twm = twm / twm.sum()
        word['maxtopic'] = twm.T.idxmax()
        self.put_table(word, 'word', index_label='word_id')

    def add_maxtopic_to_doc(self):
        """Add idmax topic for each doc"""
        # todo: Put this in the method that creates doctopic
        # doctopic = self.get_table('doctopic', set_index=True)
        doc = self.get_table('doc')
        doc = doc.set_index('doc_id')
        doc = doc.sort_index()
        sql = """
        SELECT doc_id, topic_id as maxtopic, MAX(topic_weight) as maxweight
        FROM doctopic
        GROUP BY doc_id
        """
        doc['maxtopic'] = pd.read_sql_query(sql, self.conn, index_col='doc_id').sort_index()
        # doc['maxtopic'] = doctopic.topic_weight.unstack().fillna(0).T.idxmax()
        self.put_table(doc, 'doc', index=True)

    def add_doctopic_weight_stats(self):
        """Add doctopic weight stats to config table"""
        doctopic = self.get_table('doctopic')
        items = dict(
            doctopic_weight_min=doctopic.topic_weight.min(),
            doctopic_weight_max=doctopic.topic_weight.max(),
            doctopic_weight_avg=doctopic.topic_weight.mean()
        )
        self.set_config_items(items)

    def add_doctopic_entropy_stats(self):
        """Add doctopic entropy stats to config table"""
        doc = self.get_table('doc')
        items = dict(
            doctopic_entropy_min=doc.topic_entropy.min(),
            doctopic_entropy_max=doc.topic_entropy.max(),
            doctopic_entropy_avg=doc.topic_entropy.mean()
        )
        self.set_config_items(items)

    def add_topiccompcorr(self):
        """Add topic component correlation table"""
        corpus_db_file = self.config.generate_corpus_db_file_path()
        corpus = PoloDb(corpus_db_file)
        pca_doc = corpus.get_table('pca_doc')
        del(corpus)
        pca_doc = pca_doc.set_index('doc_id')
        sql = """
        SELECT a.src_doc_id AS doc_id, topic_id, topic_weight  
        FROM doc a 
        JOIN doctopic b USING(doc_id)
        """
        doctopic = pd.read_sql_query(sql, self.conn, index_col=['doc_id', 'topic_id'])
        dtm = doctopic.unstack()
        dtm.columns = dtm.columns.droplevel(0)
        # dtm.columns = ["T{0}".format(col) for col in  dtm.columns]
        X = dtm.T.dot(pca_doc)
        self.put_table(X, 'topiccomp_corr', index=True)

        # Add topic poles
        A = X.idxmax()
        B = X.idxmin()
        C = pd.concat([A,B], 1)
        C.columns = ['max_pos_topic_id','max_neg_topic_id']
        C.index = [int(idx.replace('PC','')) for idx in C.index]
        C.index.name  = 'pc_id'
        self.put_table(C, 'topiccomp_pole', index=True)

    def add_topic_clustering(self):
        """Apply Ward clustering of topics based on topicword matrix"""
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import pdist
        
        tw = self.get_table('topicword')
        twm = tw.set_index(['word_id', 'topic_id']).unstack().fillna(0)
        twm = twm / twm.sum()
        twm.columns = twm.columns.droplevel(0)
        twm = twm.T

        topics = self.get_table('topic')
        topics['label'] = topics.apply(lambda x: "{1} T{0:02d}".format(x.name, x.topic_gloss).strip(), 1) 

        # Create plots
        import plotly.figure_factory as ff
        fig = ff.create_dendrogram(twm, orientation='left', labels=topics.label.tolist(),
            distfun=lambda x: pdist(x, metric='euclidean'),
            linkagefun=lambda x: sch.linkage(x, method='ward'))
        fig.update_layout(width=650, height=25 * self.cfg_num_topics)
        fig.layout.margin.update({'l':200})
        # fig.show()    
        fig.write_image('{}-{}-dendrogram.png'.format(self.cfg_slug, self.trial_name))

        # todo: Put SVG in database 
        # fig.write_image('{}-{}-dendrogram.svg'.format(self.cfg_slug, self.trial_name))

        # Put tree data in db
        sims = pdist(twm, metric='euclidean')
        tree = pd.DataFrame(sch.linkage(sims, method='ward'), 
            columns=['clust_a','clust_b','dist_ab','n_orig_obs'])
        tree.index.name = 'iter_id'
        self.put_table(tree, 'topictree', index=True)

    def set_config_items(self, items = dict()):
        """Add config items to config table"""
        for key in items.keys():
            self.set_config_item(key, items[key])

    sql_config_delete = "DELETE FROM config WHERE key = ?"
    sql_config_insert = "INSERT INTO config (key, value) VALUES (?,?)"
    def set_config_item(self, key, val):
        """Insert an item in the config table"""
        self.conn.execute(self.sql_config_delete, (key,))
        self.conn.execute(self.sql_config_insert, (key, val))
        self.conn.commit()

    sql_config_select = "SELECT FROM config WHERE key = ?"
    def get_config_item(self, key):
        """Get an item from the config table"""
        cur = self.conn.cursor()
        cur.execute(self.sql_config_select, (key,))
        val = cur.fetchone()[0]
        cur.close()
        return val
