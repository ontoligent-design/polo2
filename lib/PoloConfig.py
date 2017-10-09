import configparser
import os
import sys

class PoloConfig():
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

    ini_schema = {
        'DEFAULT': {
            "title": '<PROJECT TITLE>',
            "slug": '<SHORT TITLE (NO SPACES)>',
            "owner": '<PROJECT OWNER NAME>',
            "base_path": '<BASE_PATH>',
            "mallet_path": '/usr/local/bin/mallet',
            "mallet_out_dir": 'trials',
            "mallet_corpus_input": 'corpus/corpus.csv',
            "extra_stops": 'corpus/extra-stopwords.txt',
            "replacements": 'corpus/replacements.txt',
            "num_threads": 1,
            "verbose": 0,
            "nltk_data_path": '<NLTK_DATA_PATH>',
            "corpus_sep": ',',
            "use_nltk": 1,
            "use_stopwords": 1,
            "thresh": 0.05
    }, 
        'trial1': {
            "num_topics": 20,
            "num_top_words": 10,
            "num_iterations": 500,
            "optimize_interval": 10,
            "num_threads": 1
        }
    }
    
    def __init__(self, ini_file):
        self.ini_file = ini_file
        self.ini = configparser.ConfigParser()
        self.ini._interpolation = configparser.ExtendedInterpolation()
        self.ini.read(ini_file)
        self.validate_ini()
        self.trials = self.ini.sections()

    def get_trial_names(self):
        return self.trials

    def validate_ini(self):
        keys1 = self.ini_schema['DEFAULT'].keys()
        keys2 = self.ini['DEFAULT'].keys()
        test1 = self.compare_keys(keys1, keys2)
        if test1:
            print("Missing config DEFAULT keys:", ', '.join(test1))
            sys.exit(1)
        keys3 = self.ini_schema['trial1'].keys()
        for trial in self.ini.sections():
            keys4 = self.ini[trial].keys()
            test2 = self.compare_keys(keys3, keys4)
            if test2:
                print("Missing config keys for trial `{}`.".format(trial), ', '.join(test2))
                sys.exit(1)
        print("INI file `{}` seems OK".format(self.ini_file))
        return True

    def compare_keys(self, keys1, keys2):
        keys1 = set(keys1)
        keys2 = set(keys2)
        if keys1.issubset(keys2):
            return None
        else:
            diff = keys1.difference(keys2)
            return diff

    def create_ini(self, ini_file = 'config.template.ini'):
        new_ini = configparser.ConfigParser()
        new_ini.read_dict(self.ini_schema)
        if not os.path.isfile(ini_file):
            print('Creating', ini_file)
            with open(ini_file, 'w+') as configfile:
                new_ini.write(configfile)
            if os.path.isfile(ini_file):
                print("`{}` created successfully.".format(ini_file))
                print("Edit it and rename it to `config.ini`.")
            else:
                print("Oops: `{}` not created successfully.".format(ini_file))
        else:
            print("`{}` already exists.".format(ini_file))
            sys.exit(1)