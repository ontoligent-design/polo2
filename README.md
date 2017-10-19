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

## Motivation

Create a tool to simplify the process of generating models from texts and, more important, of producing interactive data products with these models.

## Approach

Put everything in the database. 

## Class Specific Guidelines

### PoloSource

#### Synopsis

```python
from polo2 import PoloSource

# Import the file into a dataframe
frankenstein = PoloSource('frankenstein.txt')
frankenstein.text.head()

# Remove cruft
pat_begin = r'^Letter\s+1\s*$'
pat_end = r'^End of the Project Gutenberg EBook'
frankenstein.clip_text(remove_blanks=False, pat_begin=pat_begin, pat_end=pat_end)

# Convert line breaks into paragraph markers
break_num = 0
def add_breaks(line):
    global break_num
    break_num += 1
    return 'BREAK {}'.format(break_num)
frankenstein.text.loc[frankenstein.text.line == '', 'line'] = frankenstein.text.line.apply(add_breaks)
frankenstein.text.head()

# Create a multicolumn data frame with sections as
# indexes
sec_pats = []
sec_pats.append(r'^\s*(?:Letter|Chapter)\s+\d+')
sec_pats.append(r'^\s*BREAK \d+')
frankenstein.unstack_text(sec_pats=sec_pats)
frankenstein.text.head()

# Convert the text dataframe into a corpus dataframe
frankenstein.text_as_corpus()
frankenstein.corpus.head()

# Export the corpus to the database
# TBD

```

Use PoloSource to download individual text documents, such as books from Project Gutenberg. These documents will generally have a basic sequential structure that can be defined as a series section breaks, each of which can be identified by a regular expression. Typically these breaks consist of the beginning of the text, the end of the text, and the text's major sections, such as book and chapter.

To use PoloSource, first create an object by passing a filename and optionally a URL:

```python
from polo2 import PoloSource

frankenstein = PoloSource('frankenstein.txt')
```

This creates a dataframe `text` from the source file, which can be inspected like so:

```python
frankenstein.text.head()
```
To remove extraneous front and backmatter, identify the lines that uniquely identify these breaks in the document and then pass these as arguments to the `clip_text()` method:

```python
pat_begin = r'^Letter\s+1\s*$'
pat_end = r'^End of the Project Gutenberg EBook'
frankenstein.clip_text(remove_blanks=False, pat_begin=pat_begin, pat_end=pat_end)
``` 
This will alter in place the `text` dataframe. To chunk the text by section, identify the section break lines and pass their regular expressions to the `unstack_text()` method. In this case, we want to use line breaks to identity paragraphs, so we modify the `text` by adding explicit labels to each break:

```python
break_num = 0
def add_breaks(line):
    global break_num
    break_num += 1
    return 'BREAK {}'.format(break_num)

frankenstein.text.loc[frankenstein.text.line == '', 'line'] = frankenstein.text.line.apply(add_breaks)
```

Then we pass these patterns to the `unstack_text()` method.

```python
sec_pats = []
sec_pats.append(r'^\s*(?:Letter|Chapter)\s+\d+')
sec_pats.append(r'^\s*BREAK \d+')
frankenstein.unstack_text(sec_pats=sec_pats)
```



