import configparser, os
from collections import OrderedDict


class PoloConfig():

    # todo: Replace schemas with XML so you can have validation.
    ini_schema = OrderedDict([
        ('DEFAULT', OrderedDict([
            ("title", 'Replace me with a descriptive title of the project, like "My Project" (without the quotes)'),
            ("slug", 'Replace me with a short code name for the project, like "myproject" (without the quotes)'),
            ("owner", 'Replace me with your name and/or email address'),
            ("base_path", 'Replace me with the base path to the project directory this file is found in'),
            ("src_file_name", "corpus/corpus.csv"),
            ("src_file_sep", '|'),
            ("src_base_url", 'Replace me with URL that can be used to view corpus documents online'),
            ("src_ord_col", 'doc_label'),  # todo: TO BE DEPRECATED
            ("normalize", 1),
            ("sentiment", 1),
            ("use_stopwords", 1),
            ("extra_stops", 'corpus/extra-stopwords.txt'),
            ("use_nltk", 1),
            ("nltk_data_path", 'Replace me with the path to nltk_data'),
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
            ("show_topics_interval", 100),
            ('get_bigrams', 1),
            ('get_trigrams', 0),
            ('group_fields', 'doc_label'), # todo: TO BE DEPRECATED
            ('groups_ini_file', 'groups.ini') # todo: Add info in help about this
        ])),
        ('trial1', OrderedDict([
            ("num_topics", 20),
            ("num_iterations", 500),
            ("optimize_interval", 10)
        ]))
    ])
    group_ini_schema = OrderedDict([
        ('DEFAULT', OrderedDict([
            ('default_field', 'doc_label # This field will be displayed with topics')
        ])),
        ('doc_label', OrderedDict([
            ('slug', 'label'),
            ('title', 'Document Label')
        ]))
    ])

    # todo: Move num_iterations and optimize_interval out of trial1
    # todo: Update PoloMallet to reflect preceding
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

        # Perhaps put into a method?
        if self.ini['DEFAULT']['src_file_sep'] == 'TAB':
            self.ini['DEFAULT']['src_file_sep'] = '\t'
        elif self.ini['DEFAULT']['src_file_sep'] == '':
            self.ini['DEFAULT']['src_file_sep'] = '|'

        self.trials = self.ini.sections()

        # Import groups info
        self.groups_ini = configparser.ConfigParser()
        self.groups_ini._interpolation = configparser.ExtendedInterpolation()
        self.groups_ini.read('{}/{}'.format(self.ini['DEFAULT']['base_path'],
                                            self.ini['DEFAULT']['groups_ini_file']))

    def get_trial_names(self):
        if len(self.trials) == 0:
            raise ValueError("No trials defined in INI file.")
        return self.trials

    def set_config_attributes(self, obj, section='DEFAULT'):
        """Puts config keys and values into object as cfg_X"""
        for key in self.ini[section]:
            setattr(obj, 'cfg_{}'.format(key), self.ini[section][key])

    def validate_ini2(self, ini_schema, ini, section):
        keys1 = ini_schema[section].keys()
        keys2 = ini[section].keys()
        test1 = self.compare_keys(keys1, keys2)
        if test1:
            raise ValueError("Missing config {} keys:".format(section), ', '.join(test1))
        return True

    # todo: To be replaced by validate_ini2()
    def validate_ini(self):
        # Well-formed test for [DEFAULT]
        keys1 = self.ini_schema['DEFAULT'].keys()
        keys2 = self.ini['DEFAULT'].keys()
        test1 = self.compare_keys(keys1, keys2)
        if test1:
            raise ValueError("Missing config DEFAULT keys:", ', '.join(test1))
        # Well-formed test for [trial1]
        # todo: Change this if we decide to put some of these keys into DEFAULT
        keys3 = self.ini_schema['trial1'].keys()
        for trial in self.ini.sections():
            keys4 = self.ini[trial].keys()
            test2 = self.compare_keys(keys3, keys4)
            if test2:
                raise ValueError("Missing config keys for trial `{}`.".format(trial), ', '.join(test2))
        # todo: Create interactive validation of ALL keys here
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

    def generate_corpus_db_file_path(self):
        corpus_db_file_path = '{}/{}-corpus.db'.format(self.ini['DEFAULT']['base_path'], self.ini['DEFAULT']['slug'])
        return corpus_db_file_path

    def generate_model_db_file_path(self, trial_name = 'trial1'):
        self.validate_trial_name(trial_name)
        corpus_db_file_path = '{}/{}-mallet-{}.db'.format(self.ini['DEFAULT']['base_path'], self.ini['DEFAULT']['slug'], trial_name)
        return corpus_db_file_path

    def validate_trial_name(self, trial_name):
        if trial_name not in self.ini:
            raise ValueError("Trial name not found")

    def get_group_fields(self):
        group_fields = [group_field.strip() for group_field in self.ini['DEFAULT']['group_fields'].split(',')]
        return group_fields