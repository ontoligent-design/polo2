# Polo2

## Getting Started

Follow the directions in the comments to get started with Polo. This will install
the package and create a project directory where you can run the polo command line
tool.

```bash
# Clone the repo somewhere suitable.
git clone https://github.com/ontoligent-design/polo2

# Change directory into the repo.
cd polo2

# Install the package. Do this every time you pull from the repo to 
# make the changes active in your python environment.
sudo python3 setup.py install

# Test the installation -- this should return a help screen.
# If this does not work, you may need to install some Python libraries,
# or you may be using the wrong version of Python.
polo

# Now move out of the repo and create a new project directory with the polo
# command line tool. For example, you might move one level up from the cloned 
# and have a project called 'myproject'. If so, do the following: 
cd ..
polo setup myproject
cd myproject

# Edit the config.ini file to match your environment and parameters.
# Information about the purpose and contents of this file is given below.
# Of course, you don't need to use emacs for this :)
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
[`configparser`](https://docs.python.org/3/library/configparser.html) library. The structure of this format is simple: 
content is divided into sections,
indicated by bracketed labels on new lines, followed by a series of key/value pairs separated by 
a colon or equal sign. If you're interested in using this format for your own projects, see the documentation to
learn about what it can do -- Polo usings only a fraction of its functionality.

When you run `polo setup myproject`, a `config.ini` file is automatically created for you, with
some values automatically added. Here is a template of this file, with comments to explain what 
each key means. Note that although this file looks long, most of it is written only once, and only
codifies things that you would have to know or enter anyway. Once written, you will be adding
only new trials, which are only three lines long each.

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

# Whether to use Polo's default normalization filters, which lowercase words and remove punction and numbers
normalize = 1

# Whether to use stopwords in further normalizing your corpus for any models you wish
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
nltk_data_path = /opt/nltk_data

# The path to MALLET on your system.
mallet_path = /usr/local/bin/mallet

# The relative path to the directory where MALLET will dump its data files. 
# This was created for you and you should not ever need to change this.
mallet_out_dir = trials

# The path to the file that polo corpus creates for mallet to use. Again, Polo
# creates this for you and you should not need to change it.
mallet_corpus_input = trials/mallet-corpus.csv

# The default number of threads for MALLET to use. 
num_threads = 1

# The default verbose setting for MALLET to use.
verbose = 0

# The default threshhold to use for determining whether a topic is said to exist in a document.
thresh = 0.05

# The relative path to replacements for MALLET, as defined by David Mimno
# here http://www.mimno.org/articles/phrases/. Basically, you can enter, one per line,
# a phrase, a tab, and the phrase's single token replacement, such as version of the 
# phrase with underlines instead of spaces. 
replacements = corpus/replacements.txt

# Default number of top words to display per topic.
num_top_words = 10

# The number of top documents to write. Can be overridden. NOT CURRENTLY USED.
num_top_docs = 100

# The number of topic proportions per documents to print, A negative value indicates that 
all topics should. 
doc_topics_max = 10

# The number of iterations after wich MALLET will print out the current set of topics. 
show_topics_interval = 100

# This is an example "trial," that is, a set of topic model hyperparameters with which to 
# run MALLET. You can create as many trial entries in the config file as you want. Each entry
# should have a unique name. The name in the header can then be passed to polo 
# in the form "polo mallet mytrail". Polo will then run MALLET and generate a SQLite database 
# for each trial you run. Databases for the same trial entry will overwrite a previous instance.
# Note that "trial1" is considered the default trial, and will run if you do not give
# polo a trial name to run. It's best not to delete this entry, but to configure it to
# suit your needs.
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

To get started running polo to create a corpus and model database, you need to create a corpus in 
a format that Polo can ingest. For consistency, that format is called "Standard Source Corpus Format,"
or SSCF for short.

> In text analytics, the word "corpus" refers to a collection of "documents," which are just
segments of text. A document does not have to correspond to a document as normally 
understand, that is, a stand-alone text of some kind, like a letter or a news article. Instead,
in can be any segment of text that is analytically useful, from a tweet to a collection of tweets, 
a paragraph or a page in book. 

SSCF has the following requirements:

* It must be in so-called CSV format, with a delimiter as defined in `config.ini` as explained in the sample above
* A header row with the following column names:
   * `doc_key`: A unique identifier for the document. We do not use `doc_id`, as this value
   is generated by Polo.
   * `doc_label`: A label to associate with the documents. This can be used later when creating 
   heat maps of topic weights per label. This can be the year of publication, or some keyword 
   associated with it. If you can't think of anything, put something like 'foo' as a placeholder.
   * `doc_title`: The title of the document.
   * `doc_content`: The content of the document to be analytically processed. Can be normalized 
   before input into Polo.
   * `doc_original`: The original format of the document, for display when exploring documents in
   an app.
   * `doc_year`: The year of the document. Optional convenience field used for display purposes.
   * `doc_date`: The date of the document. Optional convenience field used for display purposes.
*  Content consistent with the column defintions. Note that in creating this file, you can do whatever 
preprocessing you want. 

## Making use of the resulting databases

### The Corpus Database

This database stores all of the fixed (deterministic) content of the corpus, based 
on the normalization filters associated with the input corpus file and in `config.ini` 
(such as stopwords). At this point, normalization filters, such as lowercasing, are 
hard-coded in `PoloCorpus`, but this will change.

* `doc`: The document table, aka the corpus. 
    * `doc_id`: A unique ID generated by Polo.
    * `doc_key`: The document key provided by the input file.
    * `doc_year`: The year provided by the input file.
    * `doc_label`: The label provided by the input file.
    * `doc_title`: The title provided by the input file.
    * `doc_content`:  The content provided by the input file.
* `doctoken`: Tokens per document. Contains parsed tokens in order and multiple instances.
    * `doc_id`: The document containing the token.
    * `token_str`: The token str. Should really use the `token_id`. 
* `doctokenbow`: Documents and tokens as bag of words. Tokens not in order, and no duplicates.
    * `doc_id`: The document containing the token type.
    * `token_id`: The token ID.
    * `token_count`: The number of tokens of this type in the doc.
* `ngrambi`: Bigrams in the corpus.
    * `ngram`: The bigram string.
    * `ngram_count`: The number of such bigrams in the corpus.
* `ngrambidoc`: Bigrams per document.
    * `doc_id`: The containing document ID.
    * `ngram`: The bigram string.
* `stopword`: The list of stopwords.
    * `token_str`: The stopword string.
* `token`: The token table, aka "dictionary." Should really be called `token_type`.
    * `token_id`: The token ID.
    * `token_str`: The token string.
    * `token_count`: The number of tokens of this type in the corpus.

### The Model Database

* `config`: Configuration settings for downstream introspection.
    * `key`: The configuration name.
    * `value`: The configuration value.
* `doc`: The document table. Shares same ID as `corpus.doc`. Only stores IDs and topic model data.
    * `doc_id`: The document ID.
    * `doc_label`: The document label, redundant with the label in `corpus.doc` for convenience. (We try to limit such redundancies.)
    * `topic_entropy`: The topic mixture entropy in the document as calculated by MALLET.
* `doctopic`: Topics per document. 
    * `doc_id`: The document ID.
    * `topic_id`: The topic ID.
    * `topic_weight`: The weight of the topic in the document. Weights below a the threshhold (defined in `config.ini`) are considered 
    to be absent from the document. 
* `topic`: The topics generated by MALLET.
    * `topic_id`: A random ID provided by MALLET.
    * `topic_alpha`: Roughly, the frequency of the topic in the corpus.
    * `topic_words`: The top words associated with the topic in decreasing order of frequency.
    * `topic_tokens`: The number of tokens associated with the topic.
    * `topic_document_entropy`: TBD
    * `topic_word_length`: TBD
    * `topic_coherence`: TBD
    * `topic_uniform_dist`: TBD
    * `topic_corpus_dist`: TBD
    * `topic_eff_num_words`: TBD
    * `topic_token_doc_diff`: TBD
    * `topic_rank_1_docs`: TBD
    * `topic_allocation_ratio`: TBD
    * `topic_allocation_count`: TBD
    * `topic_exclusivity`: TBD
    * `topic_freq`: The raw frequency of the topic in the corpus. The number of times the topic appears in a document with a weight 
    above the threshold.
    * `topic_rel_freq`: The relative frequency of the topic in the corpus.
* `topicpair`: All topic combinations, without reverse directions.
    * `topic_a_id`: The topic ID of the first item in the pair.
    * `topic_b_id`: The topic ID of the second item in the pair.
    * `p_ab`: The joint relative frequency of the two topics in the corpus. 
    * `p_aGb`: The conditional relative frequency of the first topic given the second.
    * `p_bGa`: The conditional relative frequency of the second topic given the first.
    * `i_ab`: The adjusted pointwise mutual information between the two topics. 
    * `c_ab`: The confidence measure (in association rule theory).
    * `cosine_sim`: The distance between the two topics based on cosine similarity.
    * `js_div`: The distance between the two topics based on Jensen-Shannon divergence.
* `topicphrase`: Topic phrases associated with each topic as detected by MALLET.
    * `topic_id`: The topic ID.
    * `topic_phrase`: The phrase string.
    * `phrase_weight`: The weight of the phrase in the topic.
    * `phrase_count`: The frequency of the phrase in the topic.
* `topicword`: Words per topic.
    * `word_id`: The word ID. Should be type or token.
    * `topic_id`: The topic ID.
    * `word_count`: The number of words of this type in the topic.
* `topicword_diag`: Diagnostics associated with the `topicword` table provided by MALLET.
    * `topic_id`: The topic ID.
    * `word_id`: The word ID.
    * `word_str`: The word string (redundant).
    * `rank`: TBD
    * `count`: The number of works of this type in the topic. (Verify this.)
    * `prob`: TBD
    * `cumulative`: TBD
    * `docs`: TBD
    * `word_length`: The length of the word.
    * `coherence`: TBD
    * `uniform_dist`: TBD
    * `corpus_dist`: TBD
    * `token_doc_diff`: TBD
    * `exclusivity`: TBD
    * `index`: TBD
* `word`: Words in the corpus. Should be really be `token_type`. Also called the "dictionary."
    * `word_id`: The word ID.
    * `word_str`: The normalized word string.

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
