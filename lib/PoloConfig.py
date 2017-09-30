import configparser
# from PoloDb import PoloDb # May add this

class PoloConfig():
    """Define more sensible defaults!"""
    slug                = 'test'
    trial               = 'my_trial'
    num_top_words       = 10
    mallet_path         = '/opt/mallet/bin/mallet'
    output_dir          = 'my_dir'
    base_path           = '.' # Not a good default
    input_corpus        = 'corpus/corpus.csv'
    extra_stops         = 'corpus/extra-stopwords.csv'
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
