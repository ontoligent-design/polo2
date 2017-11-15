import configparser, os
from collections import OrderedDict


class PoloConfig():
    
    ini_schema = OrderedDict([
        ('DEFAULT', OrderedDict([
            ("title", '<TITLE>'),
            ("slug", '<SLUG>'),
            ("owner", '<PROJECT OWNER NAME>'),
            ("base_path", '<BASE_PATH>'),
            ("src_file_name", "corpus/corpus.csv"),
            ("src_file_sep", '|'),
            ("src_base_url", '<URL_WITH_NO_TRAILING_SLASH>'),
            ("src_ord_col", 'doc_label'),
            ("use_stopwords", 1),
            ("extra_stops", 'corpus/extra-stopwords.txt'),
            ("use_nltk", 1),
            ("nltk_data_path", '<NLTK_DATA_PATH>'),
            ("mallet_path", '/usr/local/bin/mallet'),
            ("mallet_out_dir", 'trials'),
            ("mallet_corpus_input", 'trials/mallet-corpus.csv'),
            ("num_threads", 1),
            ("verbose", 0),
            ("thresh", 0.05),
            ("replacements", 'corpus/replacements.txt'),
            ("num_top_words", 10),
            ("num_top_docs", 100),
            ("doc_topics_max", 10),
            ("show_topics_interval", 100)
        ])),
        ('trial1', OrderedDict([
            ("num_topics", 20),
            ("num_iterations", 500),
            ("optimize_interval", 10)
        ]))
    ])
    # todo: Move num_iterations and optimize_interval out of trial1 and update PoloMallet to refflect this

    def __init__(self, ini_file, create=True, slug=None):
        if not os.path.isfile(ini_file):
            if create:
                self.create_ini(slug, ini_file) # Passing slug here is clunky
            else:
                raise ValueError("INI file does not exist.")
        self.ini_file = ini_file
        self.ini = configparser.ConfigParser()
        self.ini._interpolation = configparser.ExtendedInterpolation()
        self.ini.read(ini_file)
        self.validate_ini()
        self.trials = self.ini.sections()

    def get_trial_names(self):
        if len(self.trials) == 0:
            raise ValueError("No trials defined in INI file.")
        return self.trials

    def validate_ini(self):
        keys1 = self.ini_schema['DEFAULT'].keys()
        keys2 = self.ini['DEFAULT'].keys()
        test1 = self.compare_keys(keys1, keys2)
        if test1:
            raise ValueError("Missing config DEFAULT keys:", ', '.join(test1))
        keys3 = self.ini_schema['trial1'].keys()
        for trial in self.ini.sections():
            keys4 = self.ini[trial].keys()
            test2 = self.compare_keys(keys3, keys4)
            if test2:
                raise ValueError("Missing config keys for trial `{}`.".format(trial), ', '.join(test2))
        return True

    def compare_keys(self, keys1, keys2):
        keys1 = set(keys1)
        keys2 = set(keys2)
        if keys1.issubset(keys2):
            return None
        else:
            diff = keys1.difference(keys2)
            return diff

    def create_ini(self, slug, ini_file = 'config.template.ini'):
        new_ini = configparser.ConfigParser()
        new_ini.read_dict(self.ini_schema)
        new_ini['DEFAULT']['slug'] = slug
        new_ini['DEFAULT']['base_path'] = '{}/{}'.format(os.getcwd(), slug)
        if not os.path.isfile(ini_file):
            with open(ini_file, 'w+') as configfile:
                new_ini.write(configfile)
            if os.path.isfile(ini_file):
                return True
            else:
                raise ValueError("`{}` not created successfully.".format(ini_file))
        else:
            raise ValueError("`{}` already exists.".format(ini_file))
