from flask import Flask, render_template
import os, configparser
from polo2 import PoloDb, PoloConfig

app = Flask(__name__)
app.config.from_object('config')

basedir = os.path.abspath(os.path.dirname(__file__))
ini_file = '{}/app-config.ini'.format(basedir)
config =  configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read(ini_file)
#projects_dir = config.get('DEFAULT', 'projects_dir')
projects_dir = app.config['PROJECTS_DIR']

@app.route("/")
def hello():
    data = {}
    data['foo'] = app.config['FOO']
    data['projects'] = [dir for dir in os.listdir(projects_dir) if os.path.isfile('{}/{}/config.ini'.format(projects_dir, dir))]
    return render_template('home.html', **data)

@app.route("/projects/<slug>")
def projects(slug):
    data = {}
    data['slug'] = slug
    data['page_title'] = 'Project ' + slug

    pcfg = PoloConfig('{}/{}/config.ini'.format(projects_dir, slug))
    corpus = PoloDb(pcfg)
    model = PoloDb(pcfg)

    return render_template("project.html", **data)


if __name__ == '__main__':
    app.run(debug=True)