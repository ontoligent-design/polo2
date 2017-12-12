from flask import Flask, render_template
import os, configparser
import pandas as pd
from polo2 import PoloDb, PoloConfig

app = Flask(__name__)
app.config.from_object('config')

base_dir = os.path.abspath(os.path.dirname(__file__))
projects_dir = app.config['PROJECTS_DIR']

@app.route("/")
@app.route("/projects")
def hello():
    data = {}
    data['projects'] = [dir for dir in os.listdir(projects_dir) if os.path.isfile('{}/{}/config.ini'.format(projects_dir, dir))]
    return render_template('home.html', **data)

@app.route('/test')
def test():
    return render_template('test.html')

@app.route("/projects/<slug>")
def projects(slug):
    data = {}
    data['slug'] = slug
    data['page_title'] = 'Project ' + slug
    corpus = get_corpus_db(slug)
    data['doc_count'] = pd.read_sql_query('SELECT count(*) as n FROM doc', corpus.conn).n.tolist()[0]
    return render_template("project.html", **data)

# Helpers

def get_project_config(slug):
    pcfg_file = '{}/{}/config.ini'.format(projects_dir, slug)
    pcfg = PoloConfig(pcfg_file)
    return pcfg

def get_corpus_db(slug):
    pcfg = get_project_config(slug)
    corpus_db_file = pcfg.generate_corpus_db_file_path()
    corpus = PoloDb(corpus_db_file)
    return corpus

def get_model_db(slug, trial):
    pcfg = get_project_config(slug)
    model_db_file = pcfg.generate_model_db_file_path(trial)
    model = PoloDb(model_db_file)
    return model

if __name__ == '__main__':
    app.run(debug=True)