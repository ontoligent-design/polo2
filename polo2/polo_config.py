import configparser
import os
import sys

class PoloConfig():

    # todo: Consider just saving this as a template file
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
            "corpus_header": '',
            "use_nltk": 1,
            "use_stopwords": 1,
            "thresh": 0.05,
            "src_script": "importer.py",
            "src_base_url": '',
            "src_ord_col": 'doc_label',
    }, 
        'trial1': {
            "num_topics": 20,
            "num_top_words": 10,
            "num_iterations": 500,
            "optimize_interval": 10,
            "num_threads": 1
        }
    }
    
    def __init__(self, ini_file, create=True):
        if not os.path.isfile(ini_file):
            if create:
                self.create_ini(ini_file)
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

    def create_ini(self, ini_file = 'config.template.ini'):
        # todo: If using a template, pass slug name as argument
        new_ini = configparser.ConfigParser()
        new_ini.read_dict(self.ini_schema)
        if not os.path.isfile(ini_file):
            with open(ini_file, 'w+') as configfile:
                new_ini.write(configfile)
            if os.path.isfile(ini_file):
                return True
            else:
                raise ValueError("`{}` not created successfully.".format(ini_file))
        else:
            raise ValueError("`{}` already exists.".format(ini_file))
