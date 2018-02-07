import sys, os.path

class PoloFile():

    def __init__(self, file_name = None):
        if os.path.isfile(file_name):
            self.file = open(file_name, 'r', encoding='utf8')
            self.file_name = file_name
            self.lines = []
            self.bigline = ''
        else:
            raise ValueError("'{}' is not a file.".format(self.file_name))

    def __del__(self):
        try:
            self.file.close()
        except AttributeError as e:
            print('No file open to close.')

    def read_lines(self):
        self.file.seek(0)
        self.lines = self.file.readlines()
        return self.lines

    def read_bigline(self):
        self.file.seek(0)
        self.bigline = self.file.read()
        return self.bigline