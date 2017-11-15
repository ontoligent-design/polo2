# Polo2

## Synopsis

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
a colon. If you're interested in using this format for your own projects, see the documentation to
learn about what it can do -- Polo usings only a fraction of its functionality.

When you run `polo setup myproject`, a `config.ini` file is automatically created for you, with
some values added for you. Here is a template of this file, with comments to explain what each key means.

```ini
[DEFAULT]
title = <TITLE>
slug = myproject
owner = <PROJECT OWNER NAME>
base_path = /Users/rca2t/Dropbox/CODE/polo2-test/projects/myproject
src_file_name = corpus/corpus.csv
src_file_sep = |
src_base_url = <URL_WITH_NO_TRAILING_SLASH>
src_ord_col = doc_label
use_stopwords = 1
extra_stops = corpus/extra-stopwords.txt
use_nltk = 1
nltk_data_path = <NLTK_DATA_PATH>
mallet_path = /usr/local/bin/mallet
mallet_out_dir = trials
mallet_corpus_input = trials/mallet-corpus.csv
num_threads = 1
verbose = 0
thresh = 0.05
replacements = corpus/replacements.txt
num_top_words = 10
num_top_docs = 100
doc_topics_max = 10
show_topics_interval = 100

[trial1]
num_topics = 20
num_iterations = 500
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
