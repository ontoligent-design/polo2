import os, sys, sqlite3, time, re
import pandas as pd
from lxml import etree
import PoloMath as pm

class PoloConfig:
    """Paramaters to be passed to mallet as well as other things."""
    """Define more sensible defaults."""
    slug                = 'test'
    trial               = 'my_trial'
    num_top_words       = 10
    mallet_path         = '/opt/mallet/bin/mallet'
    output_dir          = 'my_dir'
    base_path           = '.' # Not a good default
    input_corpus        = 'corpus.csv'
    extra_stops         = 'extra-stopwords.csv'
    replacements        = ''
    num_topics          = 20
    num_iterations      = 100
    optimize_interval   = 10
    num_threads         = 1
    verbose             = False
    thresh              = 0.05 # Used for calculating PWMI

    def __init__(self):
        self.ini = {}
        self.trials = []

    def read_ini(self, ini_file):
        import configparser
        self.ini = configparser.ConfigParser()
        self.ini._interpolation = configparser.ExtendedInterpolation()
        self.ini.read(ini_file)

    def get_trial_names(self):
        return self.ini.sections()

    def import_ini(self, trial):
        """Import config from local ini file. Handle default
        case when no trial given."""
        self.trial = trial
        self.slug = self.ini['DEFAULT']['slug']
        self.mallet_path = self.ini['DEFAULT']['mallet_path']
        self.output_dir = self.ini['DEFAULT']['mallet_out_dir']
        self.base_path = self.ini['DEFAULT']['base_path']
        self.input_corpus = self.ini[trial]['mallet_corpus_input']
        self.num_topics = self.ini[trial]['num_topics']
        self.num_iterations = self.ini[trial]['num_iterations']
        self.extra_stops = self.ini[trial]['extra_stops']
        self.replacements = self.ini[trial]['replacements']
        self.verbose = False


class PoloMallet:
    
    def __init__(self, config):

        self.config = config
        self.generate_trial_name()
        self.file_prefix = '{}/{}'.format(self.config.output_dir, self.config.trial_name)
        self.config.num_topics = int(self.config.num_topics)
        self.mallet = {'import-file': {}, 'train-topics': {}}
        self.mallet_init()
        self.dbfile = "{}/{}-{}.db".format(self.config.base_path, self.config.slug, self.config.trial)
        try:
            self.conn = sqlite3.connect(self.dbfile)
        except sqlite3.Error as e:
            print("Can't connect to database:", e.args[0])
            sys.exit(0)

    def __del__(self):
        try:
            self.conn.close()
        except sqlite3.Error as e:
            print("Can't close database:", e.args[0])

    def generate_trial_name(self):
        ts = time.time()
        self.config.trial_name = '{}-model-t{}-i{}-{}'.format(self.config.trial, self.config.num_topics,
                                                              self.config.num_iterations, int(ts))
    
    def mallet_init(self):
        if not os.path.exists(self.config.mallet_path):
            print('OOPS Mallet cannot be found')
            sys.exit(0)

        if os.path.exists(self.config.extra_stops):
            self.mallet['import-file']['extra-stopwords'] = self.config.extra_stops
        if os.path.exists(self.config.replacements):
            self.mallet['import-file']['replacement-files'] = self.config.replacements
        self.mallet['import-file']['input'] = self.config.input_corpus
        self.mallet['import-file']['output'] = '{}/{}-corpus.mallet'.format(self.config.output_dir, self.config.trial)
        self.mallet['import-file']['keep-sequence'] = 'TRUE'
        self.mallet['import-file']['remove-stopwords'] = 'TRUE'

        self.mallet['train-topics']['num-topics'] = self.config.num_topics
        self.mallet['train-topics']['num-top-words'] = self.config.num_top_words
        self.mallet['train-topics']['num-iterations'] = self.config.num_iterations
        self.mallet['train-topics']['optimize-interval'] = self.config.optimize_interval
        self.mallet['train-topics']['num-threads'] = self.config.num_threads
        self.mallet['train-topics']['input'] = self.mallet['import-file']['output']

        self.mallet['train-topics']['output-topic-keys']        = '{}-topic-keys.txt'.format(self.file_prefix)
        self.mallet['train-topics']['output-doc-topics']        = '{}-doc-topics.txt'.format(self.file_prefix)
        self.mallet['train-topics']['word-topic-counts-file']   = '{}-word-topic-counts.txt'.format(self.file_prefix)
        self.mallet['train-topics']['topic-word-weights-file']  = '{}-topic-word-weights.txt'.format(self.file_prefix)
        self.mallet['train-topics']['xml-topic-report']         = '{}-topic-report.xml'.format(self.file_prefix)
        self.mallet['train-topics']['xml-topic-phrase-report']  = '{}-topic-phrase-report.xml'.format(self.file_prefix)
        self.mallet['train-topics']['diagnostics-file']         = '{}-diagnostics.xml'.format(self.file_prefix)
        #self.mallet['train-topics']['output-topic-docs']        = '{}-topic-docs.txt'.format(self.file_prefix)

        self.mallet['train-topics']['num-top-docs']             = 100 # ADD TO CONFIG
        #self.mallet['train-topics']['doc-topics-threshold']     = self.config.thresh
        self.mallet['train-topics']['doc-topics-max']           = 10 # ADD TO CONFIG
        self.mallet['train-topics']['show-topics-interval']     = 100 # ADD TO CONFIG

        self.mallet['trial_name'] = self.config.trial_name

    def mallet_run_command(self,op):
        my_args = ['--{} {}'.format(arg,self.mallet[op][arg]) for arg in self.mallet[op]]
        my_cmd = self.config.mallet_path + ' ' + op + ' ' + ' '.join(my_args)
        print(my_cmd)
        try:
            os.system(my_cmd)
        except:
            print('Command would not execute:', my_cmd)
            sys.exit(0)

    def mallet_import(self):
        """Consider option of using previously generated file."""
        self.mallet_run_command('import-file')

    def mallet_train(self):
        self.mallet_run_command('train-topics')

    def clean_up(self):
        file_mask = '{}/{}-*.*'.format(self.config.output_dir,self.config.trial_name)
        my_cmd = 'rm {}'.format(file_mask)
        try:
            os.system(my_cmd)
        except:
            print('Unable to delete files: {}'.format(file_mask))

    # TABLE IMPORT METHODS

    def tables_to_db(self):
        self.import_table_topic()
        self.import_tables_topicword_and_word()
        self.import_table_doctopic()
        self.import_table_topicphrase()
        self.import_table_config()

    def import_table_topic(self, src_file=None):
        if not src_file: src_file = self.mallet['train-topics']['output-topic-keys']
        topic = pd.read_csv(src_file, sep='\t', header=None)
        topic.rename(columns={0:'topic_id', 1:'topic_alpha', 2:'topic_words'}, inplace=True)
        self.df_to_db(topic, 'topic')

    def import_tables_topicword_and_word(self, src_file=None):
        if not src_file: src_file = self.mallet['train-topics']['word-topic-counts-file']
        WORD = []
        TOPICWORD = []
        with open(src_file, 'r') as src:
            for line in src:
                row = line.strip().split(' ')
                (word_id, word_str) = row[0:2]
                WORD.append((int(word_id), word_str))
                for item in row[2:]:
                    (topic_id, word_count) = item.split(':')
                    TOPICWORD.append((int(word_id), int(topic_id), int(word_count)))
        word = pd.DataFrame(WORD, columns=['word_id', 'word_str'])
        topicword = pd.DataFrame(TOPICWORD, columns=['word_id', 'topic_id', 'word_count'])
        self.df_to_db(word, 'word')
        self.df_to_db(topicword, 'topicword')

    def import_table_doctopic(self, src_file=None):
        if not src_file: src_file = self.mallet['train-topics']['output-doc-topics']
        if 'doc-topics-threshold' in self.mallet['train-topics']:
            DOCTOPIC = []
            with open(src_file, 'r') as src:
                next(src) # Skip header -- BUT THIS IS A CLUE
                for line in src:
                    row = line.split('\t')
                    row.pop() # Pretty sure this is right
                    doc_id = row[0]
                    for i in range(2, len(row), 2):
                        topic_id = row[i]
                        topic_weight = row[i+1]
                        DOCTOPIC.append([doc_id, topic_id, topic_weight])
            doctopic = pd.DataFrame(DOCTOPIC, columns=['doc_id', 'topic_id', 'topic_weight'])
            self.df_to_db(doctopic, 'doctopic')
        else:
            doctopic = pd.read_csv(src_file, sep='\t', header=None)
            doctopic.drop(1, axis = 1, inplace=True)
            doctopic.rename(columns={0:'doc_id'}, inplace=True)
            y = [col for col in doctopic.columns[1:]]
            doctopic_narrow = pd.lreshape(doctopic, {'topic_weight': y})
            doctopic_narrow['topic_id'] = [i for i in range(self.config.num_topics) for doc_id in doctopic['doc_id']]
            doctopic_narrow = doctopic_narrow[['doc_id', 'topic_id', 'topic_weight']]
            self.df_to_db(doctopic_narrow, 'doctopic')


    def import_table_topicphrase(self, src_file=None):
        if not src_file: src_file = self.mallet['train-topics']['xml-topic-phrase-report']
        TOPICPHRASE = []
        with open(src_file, 'r') as f:
            tree = etree.parse(f)
            for topic in tree.xpath('/topics/topic'):
                topic_id = int(topic.xpath('@id')[0])
                #total_tokens = topic.xpath('@totalTokens')[0]
                for phrase in topic.xpath('phrase'):
                    phrase_weight = float(phrase.xpath('@weight')[0])
                    phrase_count = int(phrase.xpath('@count')[0])
                    topic_phrase = phrase.xpath('text()')[0]
                    TOPICPHRASE.append((topic_id, topic_phrase, phrase_weight, phrase_count))
        topicphrase = pd.DataFrame(TOPICPHRASE, columns=['topic_id', 'topic_phrase',
                                                         'phrase_weight', 'phrase_count'])
        self.df_to_db(topicphrase, 'topicphrase')

    def import_table_config(self):
        cfg = {}
        cfg['trial'] = self.config.trial
        cfg['dbfile'] = self.dbfile
        cfg['thresh'] = self.config.thresh
        cfg['slug'] = self.config.slug
        cfg['num_topics'] = self.config.num_topics
        cfg['base_path'] = self.config.base_path
        cfg['file_prefix'] = self.file_prefix
        config = pd.DataFrame({'key': list(cfg.keys()), 'value': list(cfg.values())})
        with sqlite3.connect(self.dbfile) as db:
            config.to_sql('config', db, if_exists='replace')

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
        with open(src_file, 'r') as f:
            tree = etree.parse(f)
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
                    topic_id = tvals[0] # Hopefully
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
        topic = pd.DataFrame(TOPIC, columns=tkeys)
        topicword = pd.DataFrame(TOPICWORD, columns=wkeys)
        self.df_to_db(topic, 'topic_diags')
        self.df_to_db(topicword, 'topicword_diags')

    def del_mallet_files(self):
        file_keys = ['output-topic-keys', 'output-doc-topics',
                     'word-topic-counts-file', 'xml-topic-report', 'xml-topic-phrase-report']
        for fk in file_keys:
            os.remove(str(self.mallet['train-topics'][fk]))

    # UPDATE OR ADD TABLES WITH STATS
    def add_topic_entropy(self):
        """This method also creates the doc table"""
        import scipy.stats as sp
        doctopic = self.db_to_df('doctopic')

        topic_entropy = doctopic.groupby('doc_id')['topic_weight'].apply(lambda x: sp.entropy(x))
        doc = pd.DataFrame({'topic_entropy': topic_entropy})

        # Also get topic sigs for each topic
        # ONLY DO THIS IF NOT SHORT ALREADY
        #dt1 = doctopic[doctopic.topic_weight >= self.config.thresh]
        #self.df_to_db(dt1, 'doctopic_short')

        self.df_to_db(doc, 'doc', index=True)

    def create_table_topicpair(self):
        thresh = self.config.thresh
        doctopic = self.db_to_df('doctopic')
        doc_num = len(doctopic.doc_id)
        topic = self.db_to_df('topic')
        topic['topic_freq'] = [len(doctopic[doctopic.topic_id == t][doctopic.topic_weight >= thresh])
                 for t in range(self.config.num_topics)]
        topic['topic_rel_freq'] = [len(doctopic[doctopic.topic_id == t][doctopic.topic_weight >= thresh]) / doc_num
                 for t in range(self.config.num_topics)]
        doctopic_wide = doctopic.pivot(index='doc_id', columns='topic_id', values='topic_weight')
        TOPICPAIR = []
        from itertools import combinations
        for pair in list(combinations(topic.topic_id, 2)):
            a = pair[0]
            b = pair[1]
            p_a = topic.loc[a, 'topic_rel_freq']
            p_b = topic.loc[b, 'topic_rel_freq']
            p_ab = len(doctopic_wide[doctopic_wide[a] >= thresh][doctopic_wide[b] >= thresh]) / doc_num
            if p_ab == 0: p_ab = .000001 # To prevent craziness in prob calcs
            p_aGb = p_ab / p_b
            p_bGa = p_ab / p_a
            i_ab = pm.pwmi(p_a, p_b, p_ab)
            c_ab = (1 - p_a) / (1 - p_aGb)
            TOPICPAIR.append([a, b, p_a, p_b, p_ab, p_aGb, p_bGa, i_ab, c_ab])
        topicpair = pd.DataFrame(TOPICPAIR, columns=['topic_a', 'topic_b', 'p_a', 'p_b', 'p_ab',
                                                     'p_aGb', 'p_bGa', 'i_ab', 'c_ab'])
        self.df_to_db(topic, 'topic')
        self.df_to_db(topicpair, 'topicpair')

    def df_to_db(self, df, table_name='test', if_exists='replace', index=False, index_label=None):
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=index, index_label=index_label)

    def db_to_df(self, table_name=''):
        sql = 'select * from {}'.format(table_name)
        df = pd.read_sql_query(sql, self.conn)
        return df

if __name__ == '__main__':
    print('Run polo instead')