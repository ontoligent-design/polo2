# Polo2

## Synopsis

```
# Clone the repo somewhere suitable.
git clone https://github.com/ontoligent-design/polo2

# Change directory into the repo.
cd polo2

# Install the package.
sudo python3 setup.py install

# Test the installation -- this should return a help screen.
polo

# Edit the config.ini file to match your environment and parameters.
# More information about this file is given below.
emacs config.ini

# Go into the corpus directory and create a scraper that pulls content from 
# some source and creates a corpus file in standard source corpus format. 
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