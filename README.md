# Polo2

## Synopsis

```bash
# Clone the repo somewhere suitable.
git clone https://github.com/ontoligent-design/polo2

# Change directory into the repo.
cd polo2

# Install the package. Do this every time you pull from the repo to 
# make the changes active in your python environment.
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
emacs config.ini

# Go into the corpus directory and create a scraper that pulls content from 
# some source and creates a corpus file in standard source corpus format. 
# Information about the requirements of this file is given below.
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

## Scraping to Standard Source Corpus Format

## Making use of the resulting databases

## Requirements
* Python 3
* SQlite 3
* NLTK
* MALLET
* Gensim 


## Motivation

Create a tool to simplify the process of generating models from texts and, 
more important, of producing interactive data products with these models.