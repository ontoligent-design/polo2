# Polo2

## To do

* Fix problem with cached tables DONE
* Command-line interaction DONE
* File i/o handling DONE
* Schema or template for config.ini DONE
* Command to set up project DONE
* Import dates and other ordinals
* Visualization (perhaps using Bokeh)
* Write PoloGensim -- no mallet needed
* Put text functions in PoloText
    * Ngrams
    * Cleaning
    * Stopwords
* Create HDP topic models

## Introduction

Polo2 is meant to simplify Polo, which now how too many things stuck together. Polo2 should be focused on running mallet and putting the results in a sqlite file. The main value adds are:

* Simplify the running of `mallet` by using config files, one for polo itself and one or more for the projects that use polo. Also, polo will be a shell command, ideally in the path, that will run some basic arguments beyond those implied by project config files.
* Simplify the use of `mallet`'s results by putting them into a normalized relational database file using `SQLite`.

## Requirements

* SQLite
* Install `mallet`
* Run `polo` to create a project directory
* Create your own corpus file.