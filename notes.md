# TODO

* Schema or template for config.ini
* Command to set up project
* Command-line interaction
* Import dates and other ordinals
* File i/o handling
* Visualization (perhaps using Bokeh)

# Design Notes

Polo2 is meant to simplify Polo, which now how too many things stuck together. Polo2 should be focused on running mallet and putting the results in a sqlite file. The main value adds are:

* Simplify the running of `mallet` by using config files, one for polo itself and one or more for the projects that use polo. Also, polo will be a shell command, ideally in the path, that will run some basic arguments beyond those implied by project config files.
* Simplify the use of `mallet`'s results by putting them into a normalized relational database file using `SQLite`.

## Concepts

* A Project is basically a corpus.

## Requirements

* A `polo` tool that reads an `ini` file and runs `mallet` and generates a `sqlite` file for use in other applications.
* Polo expects:
  * a Polo config file
  * a project config file
  * Mallet installed
  * a corpus in mallet format
  * sqlite3 (come with python)
* Polo will have utilities to generate a Mallet corpus.
* Polo

## Items

These are the things that need building

### Config files

* Project config files are called `config.ini` and should be in the root directory of the project
* There should also be a config file for Polo itself, but where to put that?

### Polo tool

* `polo` should be put in `$PATH` or symlinked in `/usr/local/bin`
* `PoloMallet` should be somewhere like `/Users/rca2t/anaconda/lib/python3.5/site-packages`

### SQLite schema

* No need to define at this point. Just dump dataframes to sql.