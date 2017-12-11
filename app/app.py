from flask import Flask
import os, configparser, sys

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
ini_file = '{}/app-config.ini'.format(basedir)
config =  configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read(ini_file)

@app.route("/")
def hello():
    projects_dir = config.get('DEFAULT', 'projects_dir')
    dirs = '<br />\n'.join([dir for dir in os.listdir(projects_dir) if os.path.isfile(dir + '/config.ini')])
    return "Projects Directory: {}".format(dirs)

@app.route("/project")
def projects():
    return "Projects"


if __name__ == '__main__':
    app.run(debug=True)