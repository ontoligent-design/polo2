# Polo2

## Getting Started

Follow the directions in the comments to get started with Polo2. This will install
the package and create a project directory where you can run the polo command line
tool.

```bash
# Clone the repo somewhere suitable.
git clone https://github.com/ontoligent-design/polo2

# Change directory into the repo.
cd polo2

# Install the package. Do this every time you pull from the repo to 
# make the changes are active in your python environment.
sudo python3 setup.py install

# Test the installation -- this should return a help screen.
# If this does not work, you may need to install some python libraries,
# or you may be using the wrong version of python.
polo

# Now move out of the repo and create a new project directory. For example,
# you might move one level up from the cloned and have a project called
# 'myproject'. If so, do the following: 
cd ..
polo setup myproject
cd myproject

# Edit the config.ini file to match your environment and parameters.
# Information about the purpose and contents of this file is given below.
# Of course, you don't need to use emacs for this. 
emacs config.ini

# Go into the corpus directory and create a scraper that pulls content from 
# some source and creates a corpus file in standard source corpus format (SSCF). 
# Information about the requirements of SSCF is given below. If you already 
# have a  corpus file or collection of files, write a script to convert it or them
# into SSCF.
cd corpus
emacs scraper.py

# Change directory back into the root of your project.
cd ..

# Make sure config.ini points to the correct source corpus file.
# Run polo corpus to generate corpus data.
polo corpus

# Run polo mallet to generate a topic model.
# Running polo mallet by itself will use parameters from
# the trail1 section of config.ini.
polo mallet

# You should now have two SQLite databases with all of the corpus and model
# data you need to run analytics and visualizations.

```
## Understanding and Editing `config.ini`

The `config.ini` is an essential part of Polo. Consistent with Polo's design philosophy (see below),
this file stores as much information as possible about the environment and parameters needed to 
run Polo, and it is passed to most Polo objects you create.  

> The file uses the simple `INI` format, support for which is conveniently built into Python, via the
[`configparser`](https://docs.python.org/3/library/configparser.html) library. The structure of this format is simple: content is divided into sections,
indicated by bracketed labels on new lines, followed by a series of key/value pairs separated by 
a colon or equal sign. If you're interested in using this format for your own projects, see the documentation to
learn about what it can do -- Polo usings only a fraction of its functionality.

When you run `polo setup myproject`, a `config.ini` file is automatically created for you, with
some values automatically added. Here is a template of this file, with comments to explain what 
each key means.

```ini
[DEFAULT]

# A short title for your project, e.g. My Project
title = <TITLE> 

# A code-like name for your project, with no spaces or 
# special characters other than numbers and an underscore.
slug = myproject 

# Your email address or your name.
owner = person@place.org 

# The location of this file and project; added for you when you run "polo setup".
base_path = /Users/rca2t/Dropbox/CODE/polo2-test/projects/myproject

# The relative path of your corpus data, which you will need to create
# and format in standard source corpus format (SSCF).
src_file_name = corpus/corpus.csv

# The delimitter used in the corpus file. For tab, don't use \t, as shown below.
# Delimitters are passed to Pandas' `DataFrame.read_csv()` method, so admissable 
# strings must be single characters AFAIK. 
src_file_sep = TAB

# The base URL of the documents in your corpus if they come from a website, and
# can be referenced by combining this URL with a suffix of some kind. 
src_base_url = http://somewordpresssite.com/?p=

# The column in your source document that will be used to visualized topic trends.
# This is likely to change in future versions of Polo.
src_ord_col = doc_label

# Whether to use stopwords in normalizing your corpus for any models you wish
# to generate.
use_stopwords = 1

# The path of a file that contains extra stopwords you may want to add to the list.
# You can update this before re-running a topic model to remove noisy words.
extra_stops = corpus/extra-stopwords.txt

# Whether to use NLTK's stopwords. If you don't, and you want to use stopwords, you
# must use the previous file for all of your stopwords.
use_nltk = 1

# The path to where you installed NLTK's data files. See NLTK for instructions on
# how to download these files.
nltk_data_path = <NLTK_DATA_PATH>

# The path to MALLET on your system.
mallet_path = /usr/local/bin/mallet

# The relative path to the directory where MALLET will dump its data files. 
# This was created for you and you should not ever need to change this.
mallet_out_dir = trials

# The path to the file that polo corpus creates for mallet to use. Again, Polo
# creates this for you and you should not need to change it.
mallet_corpus_input = trials/mallet-corpus.csv

# The default number of threads for MALLET to use. Can be overridden in a trial section 
# (see below).
num_threads = 1

# The default verbose setting for MALLET to use. Can be overridden in a trial section
verbose = 0

# The default threshhold to use for determining whether a topic is said to exist in a document.
thresh = 0.05

# The relative path to replacements for MALLET, as defined by David Mimno
# here http://www.mimno.org/articles/phrases/. Basically, you can enter, one per line,
# a phrase, a tab, and the phrase's single token replacement, such as version of the 
# phrase with underlines instead of spaces. 
replacements = corpus/replacements.txt

# Default number of top words to display per topic. Can be overridden.
num_top_words = 10

# The number of top documents to write. Can be overridden. NOT CURRENTLY USED.
num_top_docs = 100

# The number of topic proportions per documents to print, A negative value indicates that 
all topics should. Can be overridden.
doc_topics_max = 10

# "This option turns on hyperparameter optimization, which allows the model to better fit 
# the data by allowing some topics to be more prominent than others. Optimization every 10 
# iterations is reasonable." Can be overridden per trial.
show_topics_interval = 10

# This is an example trial. You can have as many as you want of these. Each trial
# should have a unique name. The name in the header can then be passed to polo mallet 
# in the form "polo mallet mytrail". Polo will generate a SQLite database for each trial
# type you run. Databases for the same trial will overwrite a previous instance.
[trial1]

# The number of topics for the model to produce. Values between 20 and 100 are common, 
# although some corpora may need up to 200. But this is rare.
num_topics = 20

# The number of iterations to generate the model. 500 is a good minimum and 1000 a good
# max. Beyond this, more is not necessarily better.
num_iterations = 500

# The number of iterations between reestimating dirichlet hyperparameters, which allows 
# the model to better fit the data by allowing some topics to be more prominent than others. 
# Optimization every 10 iterations is reasonable.
optimize_interval = 10
```

## Scraping to Standard Source Corpus Format

## Making use of the resulting databases

## Requirements
* Python 3
    * Polo is written in Python 3.6, but earlier versions of 3 should work. If you don't 
    have Python installed already, I recommend the [Anaconda distribution](https://www.anaconda.com/), even though
    `conda`, the package manager it comes with, has known conflicts with other package 
    managers, such as `brew` on MacOS. 
* [NLTK](http://www.nltk.org/)
    * This is used for stopwords right now but will be used by later functions. Install
    this using a package manager like `conda` or just use `pip`, but once installed,
    make sure to download `nltk_data` to get relevant resources.
* [MALLET](http://mallet.cs.umass.edu/)
    * This of course is essential. You'll need Java installed first. Once installed, you
    will need to know where the `mallet` binary is located.
* [SQlite 3](https://www.sqlite.org/)
    * You usually don't need to install this, as most unix-based OSes have it
    built in and Python has native support for it.

Optional
* [Gensim](https://radimrehurek.com/gensim/)
    * Currently, Polo has an experimental library to take advantage of Gensim's models, such as 
    Hierarchical Dirichlet Process (HDP). But you don't need to use this library to use Polo.
* [SQLiteStudio](https://sqlitestudio.pl/) or [DB Browser for SQLite](http://sqlitebrowser.org/)
    * These are free SQLite GUIs that you may find helpful in working with the data generated
    by Polo.  


## Polo's Parts

### PoloConfig

### PoloDb

### PoloCorpus

### PoloMallet

### PoloMath

### PoloGensim

## Motivation

Create a tool to simplify the process of generating models from texts and, 
more important, of producing interactive data products with these models.

## Philosophy

Polo is built on the principle that software developers should put as much information
as possible into config files and databases (the fewer of these the better) and to save logic for
essential data processing work. The idea is to remove contingent information from code and to make 
program design more solid and elegant. Functions
should be as pure as possible, with minimum side effects, and their logic should be intuitive
given the data structures they work on.  Although Polo is by no means written as a functional 
program, it strives to be functional in a general sense, and to be as interpretable as 
possible to users.
