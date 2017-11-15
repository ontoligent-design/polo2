import requests, re, os
import pandas as pd

"""
This is really just a helper for converting text files into vector format. 
"""
# todo: Consider renaming this class

class PoloText:

    #PUNC_PAT = r'\s*[.;:?!-]+\s*'
    #PUNC = re.compile(PUNC_PAT)

    def __init__(self, filename, file_url='', refresh_file=False, delim = r'\n'):
        self.filename = filename
        self.file_url = file_url
        self.text = None
        if self.file_url and (not os.path.isfile(self.filename) or refresh_file):
            self.download_file(self.file_url, self.filename)
        if not os.path.isfile(self.filename):
            raise ValueError("File `{}` not found.".format(self.filename))
        self.text = self.parse_file_as_lines(self.filename, delim=delim)

    def download_file(self, file_url=None, filename=None):
        """Downloads a file from a URL and saves it unchanged as a local file.
        The filename passed is saved as an object attribute."""
        r = requests.get(file_url)
        with open(filename, 'w') as outfile:
            outfile.write(r.text)

    def parse_file_as_lines(self, filename, delim = r'\n'):
        """Parses a local file into a dataframe with one line per row.
        Saved as the object attribute self.text."""
        with open(filename, 'r') as file:
            bigline = file.read()
            text = pd.DataFrame(re.split(delim, bigline))
            text.columns = ['line']
            return text

    def clip_text(self, pat_begin, pat_end, remove_blanks=True):
        """Removes front and backmatter of text, such as Gutenberg's boilerplate statements.
        Optionalally remove all blank lines. Alters existing text dataframe."""

        i_start = 0
        i_end = None

        if pat_begin:
            try:
                i_start = self.text[self.text.line.str.contains(pat_begin)].index.tolist()[0]
            except IndexError as e:
                print('pat_begin not matched; using', i_start)

        if pat_end:
            try:
                i_end = self.text[self.text.line.str.contains(pat_end)].index.tolist()[0]
            except IndexError as e:
                i_end = len(self.text)
                print('pat_end not matched; using', i_end)

        self.text = self.text.iloc[i_start:i_end]
        self.text.reset_index(inplace=True, drop=True)

        if remove_blanks:
            blanks = self.text.line.str.match('^\s*$')
            self.text = self.text[-blanks]

    def unstack_text(self, sec_pats = []):
        """Pass a list of regex patterns that match the various section heading lines of the text.
        For example, a top level heading might appear as 'BOOK I', in which case the pattern
        would be something like r'^\s*BOOK\s+$'. Alters existing text dataframe.
        """
        cols = ['sec'+str(i) for i in range(len(sec_pats))]
        for i, sec_pat in enumerate(sec_pats):
            secs = self.text.line.str.contains(sec_pat)
            self.text[cols[i]] = self.text[secs].line
            self.text[cols[i]].fillna(method='ffill', inplace=True)
            self.text = self.text[-secs]
        self.text.set_index(cols, inplace=True)
        self.group_cols = cols # Why not sec_cols?

    def text_as_corpus(self):
        """Create a corpus from the text, using the sections created in unstack_text()
        as the document containers."""
        self.corpus = pd.DataFrame({'doc_content': self.text.groupby(by=self.group_cols, sort=False).apply(
            lambda x: ' '.join(x.line))}).reset_index()
        self.corpus['doc_key'] = self.corpus[self.group_cols].apply(lambda x: ' | '.join(x), axis=1)
        self.corpus = self.corpus[['doc_key', 'doc_content']]

    """
    def parse_text_as_sentences(self):
        #self.text['sentence'] = self.text.apply(lambda x: self.PUNC.split(' '.join(x.line)))
        self.text['sentence'] = self.text.line.str.split(self.PUNC_PAT)
        #self.sentences = self.sentences.str.replace(',', '')
    """

"""
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
"""