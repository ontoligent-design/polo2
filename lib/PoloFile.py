import sys, os.path

class PoloFile():

    def __init__(self, file_name = None):
        if os.path.isfile(file_name):
            self.file_name = file_name
            self.lines = []
            self.bigline = ''
        else:
            print("{} is not a file.".format(self.file_name))
            sys.exit(1)

    def read_lines(self):
        with open(self.file_name, 'r') as f:
            self.lines = f.readlines()
        return(self.lines)

    def read_bigline(self):
        with open(self.file_name, 'r') as f:
            self.bigline = f.read()
        return(self.bigline)