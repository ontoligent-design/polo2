import os, sys
from flask import Flask, render_template
#from flask_caching import Cache
from polo2 import PoloDb, PoloConfig

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)
from elements import Elements

app = Flask(__name__)
app.config.from_object('config')
#cache = Cache(app,config={'CACHE_TYPE': 'simple'})

projects_dir = app.config['PROJECTS_DIR']
data = {} # Use to store variables to pass to templates

# todo: Write a Drupal-like menu handler
data['main_menu'] = {
    '/projects': 'Projects'
}

@app.route("/")
@app.route("/projects")
def hello():
    data['page_title'] = 'Project List'
    data['projects'] = [dir for dir in os.listdir(projects_dir) if os.path.isfile(get_project_config_file(dir))]
    return render_template('home.html', **data)

@app.route('/test')
def test():
    return render_template('test.html')

@app.route("/projects/<slug>")
@app.route("/projects/<slug>/<trial>")
def project(slug, trial='trial1'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)

    data['slug'] = slug
    data['trial'] = trial
    data['page_title'] = '{}, {}'.format(slug, trial)

    # todo: Find a better way to do handle menus
    path_prefix =  '/projects/{}/{}'.format(slug, trial)
    data['sub_menu'] = [
        ("{}".format(path_prefix), "Project"),
        ("{}/topic_heatmap/label".format(path_prefix), "Topic Label Heatmap"),
        ("{}/topic_heatmap/ord".format(path_prefix), "Topic Ordinal Heatmap"),
        ("{}/topic_pair_net/0.18".format(path_prefix), "Topic Pair Network"),
        ("{}/topic_pair_heatmap/jsd".format(path_prefix), "Topic Pair Similiarity Heatmap"),
        ("{}/topic_pair_heatmap/i_ab".format(path_prefix), "Topic Pair Contiguity Heatmap"),
        ("{}/docs".format(path_prefix), "Documents")
    ]

    data['ini'] = cfg.ini['DEFAULT'] # Really?
    data['trials'] = cfg.get_trial_names()
    data['src_ord_col'] = cfg.ini['DEFAULT']['src_ord_col']

    data['doc_count'] = els.get_doc_count()
    data['topic_count'] = els.get_topic_count()

    data['topics'] = els.get_topics()

    data['bigrams'] = els.get_top_bigrams()
    data['dtm'] = els.get_topicdocord_matrix()
    data['doc_ord_counts'] = els.get_topicdocgrooup_counts('topicdocord_matrix_counts')
    data['dtm_sums'] = els.get_topicdoc_sum_matrix(data['dtm'], data['doc_ord_counts'])
    return render_template("project.html", **data)

# fixme: Deprecated function
@app.route("/projects/<slug>/<trial>/topic_label_heatmap")
def topicdoc_label_heatmap(slug, trial='trial1'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['ini'] = cfg.ini['DEFAULT']
    data['trials'] = cfg.get_trial_names()
    data['slug'] = slug
    data['trial'] = trial
    data['page_title'] = '{}, {}: Topic-Label Heatmap'.format(slug, trial)
    data['dtm'] = els.get_topicdoclabel_matrix()
    return render_template("topic_label_heatmap.html", **data)

@app.route("/projects/<slug>/<trial>/topic_heatmap/<by>")
def topicdoc_heatmap(slug, trial='trial1', by='label'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['ini'] = cfg.ini['DEFAULT']
    data['trials'] = cfg.get_trial_names()
    data['slug'] = slug
    data['trial'] = trial
    data['page_title'] = '{}, {}: Topic-{} Heatmap'.format(slug, trial, by)
    data['dtm'] = els.get_topicdoc_matrix(by = by)
    return render_template("topic_label_heatmap.html", **data)

@app.route("/projects/<slug>/<trial>/topic_pair_heatmap/<sim>")
def topic_pair_heatmap(slug, trial='trial1', sim=None):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['ini'] = cfg.ini['DEFAULT']
    data['trials'] = cfg.get_trial_names()
    data['slug'] = slug
    data['trial'] = trial
    data['sim'] = sim
    data['page_title'] = '{}, {}: Topic Pair Heatmap by {}'.format(slug, trial, sim)
    data['tpm'] = els.get_topicpair_matrix()
    return render_template("topic_pair_heatmap.html", **data)

@app.route("/projects/<slug>/<trial>/topic_pair_net/<float:thresh>")
def topic_pair_net(slug, trial='trial1', thresh=0.05):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['ini'] = cfg.ini['DEFAULT']
    data['trials'] = cfg.get_trial_names()
    data['slug'] = slug
    data['trial'] = trial
    data['thresh'] = thresh
    data['page_title'] = '{}, {}: Topic Pair Network (I(a;b) >= {})'.format(slug, trial, thresh)
    data['nodes'], data['edges'] = els.get_topicpair_net(thresh)
    return render_template("topic_pair_net.html", **data)

@app.route('/projects/<slug>/<trial>/docs')
def doc_list(slug, trial='trial1'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['slug'] = slug
    data['trial'] = trial
    path_prefix =  '/projects/{}/{}'.format(slug, trial)
    data['sub_menu'] = [
        ("{}".format(path_prefix), "Project"),
        ("{}/topic_label_heatmap".format(path_prefix), "Topic Label Heatmap"),
        ("{}/topic_pair_heatmap/jsd".format(path_prefix), "Topic Pair Similiarity Heatmap"),
        ("{}/topic_pair_heatmap/i_ab".format(path_prefix), "Topic Pair Contiguity Heatmap")
    ]
    data['page_title'] = '{}, {}'.format(slug, trial)
    data['doc_entropy'] = els.get_doc_entropy()
    data['doc_entropy_avg'] = els.get_doc_entropy_avg()
    data['topic_entropy'] = data['doc_entropy_avg']
    data['docs'] = els.get_docs_for_topic_entropy(data['doc_entropy_avg'])
    return render_template('doc_list.html', **data)

@app.route('/projects/<slug>/<trial>/docs/h/<topic_entropy>')
def docs_for_entropy(slug, topic_entropy, trial='trial1'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['slug'] = slug
    data['trial'] = trial
    path_prefix =  '/projects/{}/{}'.format(slug, trial)
    data['sub_menu'] = [
        ("{}".format(path_prefix), "Project"),
        ("{}/topic_label_heatmap".format(path_prefix), "Topic Label Heatmap"),
        ("{}/topic_pair_heatmap/jsd".format(path_prefix), "Topic Pair Similiarity Heatmap"),
        ("{}/topic_pair_heatmap/i_ab".format(path_prefix), "Topic Pair Contiguity Heatmap")
    ]
    data['page_title'] = '{}, {}, Docs with Topic Entropy {}'.format(slug, trial, topic_entropy)
    data['doc_entropy'] = els.get_doc_entropy()
    data['doc_entropy_avg'] = els.get_doc_entropy_avg()
    data['topic_entropy'] = float(topic_entropy)
    data['docs'] = els.get_docs_for_topic_entropy(topic_entropy)
    return render_template('doc_list.html', **data)

@app.route('/projects/<slug>/<trial>/docs/<int:topic_id>/<doc_label>')
def docs_for_topic_and_label(slug, topic_id, doc_label, trial='trial1'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['slug'] = slug
    data['trial'] = trial
    path_prefix =  '/projects/{}/{}'.format(slug, trial)
    data['sub_menu'] = [
        ("{}".format(path_prefix), "Project"),
        ("{}/topic_label_heatmap".format(path_prefix), "Topic Label Heatmap"),
        ("{}/topic_pair_heatmap/jsd".format(path_prefix), "Topic Pair Similiarity Heatmap"),
        ("{}/topic_pair_heatmap/i_ab".format(path_prefix), "Topic Pair Contiguity Heatmap")
    ]
    data['page_title'] = '{}, {}: Topic {}, Label {}'.format(slug, trial, topic_id, doc_label)
    data['topic_id'] = topic_id
    data['doc_label'] = doc_label
    data['doc_entropy'] = els.get_doc_entropy()
    data['doc_entropy_avg'] = els.get_doc_entropy_avg()
    data['topic_entropy'] = data['doc_entropy_avg']
    data['docs'] = els.get_docs_for_topic_and_label(topic_id, doc_label)
    return render_template('doc_list.html', **data)

@app.route("/projects/<slug>/<trial>/topic/<int:topic_id>")
def topic(slug, topic_id, trial='trial1'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['topic_id'] = topic_id
    data['slug'] = slug
    data['trial'] = trial
    data['page_title'] = '{}, {}: Topic {}'.format(slug, trial, topic_id)
    data['topics'] = els.get_topics()
    data['topic'] = els.get_topic(topic_id)
    data['trend'] = els.get_topicdoc_ord_for_topic(topic_id)
    data['rels'] = els.get_topics_related(topic_id)
    data['docs'] = els.get_docs_for_topic(topic_id)
    return render_template('topic.html', **data)

# Helpers -- Consider moving to module
def get_project_config_file(slug):
    return '{}/{}/config.ini'.format(projects_dir, slug)

def get_project_config(slug):
    pcfg_file = get_project_config_file(slug)
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