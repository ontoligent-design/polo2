import os, time, re
import pandas as pd
from lxml import etree
from polo2 import PoloDb
from polo2 import PoloFile
from polo2 import PoloMath as pm


class PoloMallet(PoloDb):

    def __init__(self, config, trial):

        if trial not in config.trials:
            raise ValueError("Invalid trail name `{}`.format(trial)")

        self.config = config
        self.trial = trial

        self.cfg_slug = self.config.ini['DEFAULT']['slug']
        self.cfg_mallet_path = self.config.ini['DEFAULT']['mallet_path']
        self.cfg_output_dir = self.config.ini['DEFAULT']['mallet_out_dir']
        self.cfg_base_path = self.config.ini['DEFAULT']['base_path']
        self.cfg_verbose = self.config.ini['DEFAULT']['replacements']
        self.cfg_thresh = float(self.config.ini['DEFAULT']['thresh']) # Maybe just cast when used
        self.cfg_input_corpus = self.config.ini['DEFAULT']['mallet_corpus_input']
        self.cfg_num_top_docs = self.config.ini['DEFAULT']['num_top_docs']
        self.cfg_doc_topics_max = self.config.ini['DEFAULT']['doc_topics_max']
        self.cfg_show_topics_interval = self.config.ini['DEFAULT']['show_topics_interval']
        self.cfg_num_top_words = int(self.config.ini['DEFAULT']['num_top_words'])
        self.cfg_num_threads = int(self.config.ini['DEFAULT']['num_threads'])
        self.cfg_extra_stops = self.config.ini['DEFAULT']['extra_stops']
        self.cfg_replacements = self.config.ini['DEFAULT']['replacements']

        self.cfg_num_topics = int(self.config.ini[trial]['num_topics'])
        self.cfg_num_iterations = int(self.config.ini[trial]['num_iterations'])
        self.cfg_optimize_interval = int(self.config.ini[trial]['optimize_interval']) # Put into DEFAULT

        # fixme: Somehow this changes thresh from float to str
        # Trial overrides
        #for key in self.config.ini['DEFAULT']:
        #    if key in self.config.ini[trial]:
        #        setattr(self, 'cfg_{}'.format(key), self.config.ini[trial][key])

        self.generate_trial_name()
        self.file_prefix = '{}/{}'.format(self.cfg_output_dir, self.trial_name)
        self.mallet = {'import-file': {}, 'train-topics': {}}
        self.mallet_init()

        dbfile = "{}/{}-mallet-{}.db".format(self.cfg_base_path, self.cfg_slug, self.trial)
        PoloDb.__init__(self, dbfile)

    def generate_trial_name(self):
        ts = time.time()
        self.trial_name = '{}-model-t{}-i{}-{}'.format(self.trial, self.cfg_num_topics,
                                                              self.cfg_num_iterations, int(ts))
    
    def mallet_init(self):
        if not os.path.exists(self.cfg_mallet_path):
            raise ValueError('Mallet cannot be found.')
        #if os.path.exists(self.cfg_extra_stops):
        #    self.mallet['import-file']['extra-stopwords'] = self.cfg_extra_stops
        if os.path.exists(self.cfg_replacements): # todo: Consider moving this step out of MALLET and into corpus prep
            self.mallet['import-file']['replacement-files'] = self.cfg_replacements
        self.mallet['import-file']['input'] = self.cfg_input_corpus
        self.mallet['import-file']['output'] = '{}/{}-corpus.mallet'.format(self.cfg_output_dir, self.trial)
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
        self.mallet['train-topics']['num-top-docs'] = self.cfg_num_topics
        self.mallet['train-topics']['doc-topics-max'] = self.cfg_doc_topics_max
        self.mallet['train-topics']['show-topics-interval'] = self.cfg_show_topics_interval
        self.mallet['trial_name'] = self.trial_name

    def mallet_run_command(self, op):
        my_args = ['--{} {}'.format(arg,self.mallet[op][arg]) for arg in self.mallet[op]]
        my_cmd = '{} {} {}'.format(self.cfg_mallet_path, op, ' '.join(my_args))
        try:
            os.system(my_cmd)
        except:
            raise ValueError('Command would not execute:', my_cmd)

    def mallet_import(self):
        self.mallet_run_command('import-file')

    def mallet_train(self):
        self.mallet_run_command('train-topics')

    def clean_up(self):
        file_mask = '{}/{}-*.*'.format(self.cfg_output_dir,self.trial_name)
        my_cmd = 'rm {}'.format(file_mask)
        try:
            os.system(my_cmd)
        except:
            raise ValueError('Unable to delete files: {}'.format(file_mask))

    # TABLE IMPORT METHODS

    def tables_to_db(self):
        self.import_table_topic()
        self.import_tables_topicword_and_word()
        self.import_table_doctopic()
        self.import_table_topicphrase()
        self.import_table_config()

    def import_table_topic(self, src_file=None):
        if not src_file: src_file = self.mallet['train-topics']['output-topic-keys']
        topic = pd.read_csv(src_file, sep='\t', header=None, index_col=False,
                            names=['topic_id', 'topic_alpha', 'topic_words'])
        topic.set_index('topic_id', inplace=True)
        self.put_table(topic, 'topic')

    def import_tables_topicword_and_word(self, src_file=None):
        if not src_file: src_file = self.mallet['train-topics']['word-topic-counts-file']
        WORD = []
        TOPICWORD = []
        src = PoloFile(src_file)
        for line in src.read_lines():
            row = line.strip().split(' ')
            (word_id, word_str) = row[0:2]
            WORD.append((int(word_id), word_str))
            for item in row[2:]:
                (topic_id, word_count) = item.split(':')
                TOPICWORD.append((int(word_id), int(topic_id), int(word_count)))
        word = pd.DataFrame(WORD, columns=['word_id', 'word_str'])
        topicword = pd.DataFrame(TOPICWORD, columns=['word_id', 'topic_id', 'word_count'])
        self.put_table(word, 'word')
        self.put_table(topicword, 'topicword')

    def import_table_doctopic(self, src_file=None):
        if not src_file: src_file = self.mallet['train-topics']['output-doc-topics']
        if 'doc-topics-threshold' in self.mallet['train-topics']:
            DOC = []
            DOCTOPIC = []
            src = PoloFile(src_file)
            for line in src[1:]:
                row = line.split('\t')
                row.pop()  # Pretty sure this is right
                doc_id = row[0]
                doc_label = row[1].split(',')[1]
                DOC.append([doc_id, doc_label])
                for i in range(2, len(row), 2):
                    topic_id = row[i]
                    topic_weight = row[i + 1]
                    DOCTOPIC.append([doc_id, topic_id, topic_weight])
            doctopic = pd.DataFrame(DOCTOPIC, columns=['doc_id', 'topic_id', 'topic_weight'])
            doc = pd.DataFrame(DOC, columns=['doc_id', 'doc_label'])
            doc.set_index('doc_id', inplace=True)
            self.put_table(doctopic, 'doctopic')
            self.put_table(doc, 'doc', index=True)
        else:
            doctopic = pd.read_csv(src_file, sep='\t', header=None)
            doc = pd.DataFrame(doctopic.iloc[:, 1])
            doc.columns = ['doc_tmp']
            doc['doc_label'] = doc.doc_tmp.apply(lambda x: x.split(',')[1])
            doc = doc[['doc_label']]
            doc.index.name = 'doc_id'
            self.put_table(doc, 'doc', index=True)
            doctopic.drop(1, axis = 1, inplace=True)
            doctopic.rename(columns={0:'doc_id'}, inplace=True)
            y = [col for col in doctopic.columns[1:]]
            doctopic_narrow = pd.lreshape(doctopic, {'topic_weight': y})
            doctopic_narrow['topic_id'] = [i for i in range(self.cfg_num_topics) for doc_id in doctopic['doc_id']]
            doctopic_narrow = doctopic_narrow[['doc_id', 'topic_id', 'topic_weight']]
            self.put_table(doctopic_narrow, 'doctopic')

    def import_table_topicphrase(self, src_file=None):
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
        self.put_table(topicphrase, 'topicphrase')

    def import_table_config(self):
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
        if not src_file: src_file = self.mallet['train-topics']['diagnostics-file']
        TOPIC = []
        TOPICWORD = []
        tkeys = ['id', 'tokens', 'document_entropy', 'word-length', 'coherence', 'uniform_dist', 'corpus_dist',
                 'eff_num_words', 'token-doc-diff', 'rank_1_docs', 'allocation_ratio', 'allocation_count',
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
        topics = self.get_table('topic')
        topics.set_index('topic_id', inplace=True)
        topics = pd.concat([topics, topic_diags], axis=1)
        self.put_table(topics, 'topic', index=True) # fixme: This adds an extra index column

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
        file_keys = ['output-topic-keys', 'output-doc-topics',
                     'word-topic-counts-file', 'xml-topic-report', 'xml-topic-phrase-report',
                     'diagnostics-file', 'topic-word-weights-file']
        for fk in file_keys:
            if os.path.isfile(self.mallet['train-topics'][fk]):
                os.remove(str(self.mallet['train-topics'][fk]))

    # UPDATE OR ADD TABLES WITH STATS

    def add_topic_entropy(self):
        import scipy.stats as sp
        doctopic = self.get_table('doctopic')
        doc = self.get_table('doc')
        topic_entropy = doctopic.groupby('doc_id')['topic_weight'].apply(lambda x: sp.entropy(x))
        doc['topic_entropy'] = topic_entropy
        self.put_table(doc, 'doc')

    def create_table_topicpair(self):

        doc = self.get_table('doc')
        doc_num = len(doc.index)
        del doc

        doctopic = self.get_table('doctopic')
        dts = doctopic[doctopic.topic_weight >= self.cfg_thresh]
        dtsw = dts.pivot(index='doc_id', columns='topic_id', values='topic_weight')
        del doctopic
        del dts

        topic = self.get_table('topic')
        topic['topic_freq'] = [len(dtsw[dtsw[t] > 0]) for t in range(self.cfg_num_topics)]
        topic['topic_rel_freq'] = [len(dtsw[dtsw[t] > 0]) / doc_num for t in range(self.cfg_num_topics)]
        self.put_table(topic, 'topic')

        # For cosine sim
        topicword = self.get_table('topicword')
        topicword['word_count'] = topicword['word_count'].astype(int)
        topicword.set_index(['word_id', 'topic_id'], inplace=True)
        topicword_wide = topicword.unstack().reset_index().fillna(0)
        topicword_wide.columns = topicword_wide.columns.droplevel(0)

        TOPICPAIR = []
        from itertools import combinations
        for pair in list(combinations(topic.index, 2)):
            a = pair[0]
            b = pair[1]

            # Cosine sim and Jensen-Shannon Divergence
            x = topicword_wide.iloc[:, a].tolist()
            y = topicword_wide.iloc[:, b].tolist()
            cosim = pm.cosine_sim(x, y)
            jsdiv = pm.js_divergence(x, y)

            p_a = topic.loc[a, 'topic_rel_freq']
            p_b = topic.loc[b, 'topic_rel_freq']
            p_ab = len(dtsw[(dtsw[a] > 0) & (dtsw[b] > 0)]) / doc_num
            if p_ab == 0: p_ab = .000001 # To prevent craziness in prob calcs
            p_aGb = p_ab / p_b
            p_bGa = p_ab / p_a
            i_ab = pm.pwmi(p_a, p_b, p_ab)
            c_ab = (1 - p_a) / (1 - p_aGb)
            TOPICPAIR.append([a, b, p_ab, p_aGb, p_bGa, i_ab, c_ab, cosim, jsdiv])
        topicpair = pd.DataFrame(TOPICPAIR, columns=['topic_a_id', 'topic_b_id', 'p_ab',
                                                     'p_aGb', 'p_bGa', 'i_ab', 'c_ab', 'cosine_sim', 'js_div'])
        self.put_table(topicpair, 'topicpair')

    def get_doctopic_wide(self):
        doctopic = self.get_table('doctopic', set_index=True)
        #doctopic.set_index(['doc_id', 'topic_id'], inplace=True)
        doctopic_wide = doctopic.unstack()
        return doctopic_wide