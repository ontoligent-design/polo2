from flask import Flask, render_template
import os, configparser, sys

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
ini_file = '{}/app-config.ini'.format(basedir)
config =  configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read(ini_file)

@app.route("/")
def hello():
    data = {}
    data['projects_dir'] = config.get('DEFAULT', 'projects_dir')
    data['dirs'] = [dir for dir in os.listdir(data['projects_dir'])
                    if os.path.isfile('{}/{}/config.ini'.format(data['projects_dir'], dir))]
    return render_template('home.html', **data)

@app.route("/project")
def projects():
    return "Projects"


if __name__ == '__main__':
    app.run(debug=True)