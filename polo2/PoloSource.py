import requests, re, os
import pandas as pd
import logging

class PoloSource:

    #PUNC_PAT = r'\s*[.;:?!-]+\s*'
    #PUNC = re.compile(PUNC_PAT)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
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
            self.text = self.text.fillna(method='ffill') # This is an amazing method
            self.text = self.text[-secs]
        self.text.set_index(cols, inplace=True)
        self.group_cols = cols

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