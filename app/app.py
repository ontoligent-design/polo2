#!/usr/bin/env python

# Import installed modules
import os, sys
from flask import Flask, render_template
from polo2 import PoloDb, PoloConfig

# Import local modules
base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)
from elements import Elements, Corpus

# Create application object
app = Flask(__name__)
app.config.from_object('config')

# Define some things used in the handlers

projects_dir = os.environ.get('POLO_PROJ')
if not projects_dir:
    projects_dir = app.config['PROJECTS_DIR']
data = {} # Use to store variables to pass to templates
data['main_menu'] = {
    '/projects': 'Projects'
}

# URL Handlers

@app.route("/")
@app.route("/projects")
def hello():
    data['page_title'] = 'Project List'
    data['main_menu'] = {'/projects': 'Projects'}
    data['projects'] = {}
    for dirname in os.listdir(projects_dir):
        if os.path.isfile(get_project_config_file(dirname)):
            my_config_ini = '{}/{}/config.ini'.format(projects_dir, dirname)
            my_cfg = PoloConfig(my_config_ini)
            title = my_cfg.ini['DEFAULT']['title']
            trials = my_cfg.get_trial_names()
            data['projects'][dirname] = {'title': title, 'trials': trials}
    return render_template('home.html', **data)


@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/projects/<slug>/corpus')
def corpus(slug):
    cfg = get_project_config_file(slug)
    cps = Corpus(cfg)


@app.route("/projects/<slug>")
@app.route("/projects/<slug>/<trial>")
def project(slug, trial='trial1'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    set_project_menu(cfg, slug, trial)
    data['slug'] = slug
    data['trial'] = trial
    data['pub_home'] = os.environ['POLO_PUB']
    data['page_title'] = '{}, {}'.format(slug, trial)
    data['ini'] = cfg.ini['DEFAULT'] # Really?
    data['trials'] = cfg.get_trial_names()
    data['groups'] = cfg.get_group_fields()
    data['src_ord_col'] = cfg.ini['DEFAULT']['src_ord_col']
    data['doc_count'] = els.get_doc_count()
    data['topic_count'] = els.get_topic_count()
    data['topics'] = els.get_topics()
    data['bigrams'] = els.get_top_bigrams()
    data['phrases'] = els.get_all_topic_phrases()
    #data['ngm'] = els.get_ngram_group_matrix(degree=2)
    src_ord_col = cfg.ini['DEFAULT']['src_ord_col']
    print(src_ord_col)
    data['dtm'] = els.get_topicdoc_group_matrix(group_field=src_ord_col)
    data['doc_ord_counts'] = els.get_topicdocgrooup_counts('topic{}_matrix_counts'.format(src_ord_col))
    data['dtm_sums'] = els.get_topicdoc_sum_matrix(data['dtm'], data['doc_ord_counts'])
    return render_template("project.html", **data)

"""
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
"""


@app.route("/projects/<slug>/<trial>/topic_heatmap/<group_field>")
def topicdoc_heatmap(slug, trial='trial1', group_field='label'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    set_project_menu(cfg, slug, trial)
    data['ini'] = cfg.ini['DEFAULT']
    data['trials'] = cfg.get_trial_names()
    data['slug'] = slug
    data['trial'] = trial
    data['page_title'] = '{}, {}: Topic-{} Heatmap'.format(slug, trial, group_field)
    data['dtm'] = els.get_topicdoc_group_matrix(group_field=group_field, use_gloss_label=True)
    return render_template("topic_label_heatmap.html", **data)


@app.route("/projects/<slug>/<trial>/topic_pair_heatmap/<sim>")
def topic_pair_heatmap(slug, trial='trial1', sim=None):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    set_project_menu(cfg, slug, trial)
    data['ini'] = cfg.ini['DEFAULT']
    data['trials'] = cfg.get_trial_names()
    data['slug'] = slug
    data['trial'] = trial
    data['sim'] = sim
    data['page_title'] = '{}, {}: Topic Pair Heatmap by {}'.format(slug, trial, sim)
    data['tpm'] = els.get_topicpair_matrix()
    return render_template("topic_pair_heatmap.html", **data)


@app.route("/projects/<slug>/<trial>/topic_pair_net")
@app.route("/projects/<slug>/<trial>/topic_pair_net/<float:thresh>")
def topic_pair_net(slug, trial='trial1', thresh=0.5):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    set_project_menu(cfg, slug, trial)
    data['ini'] = cfg.ini['DEFAULT']
    data['trials'] = cfg.get_trial_names()
    data['slug'] = slug
    data['trial'] = trial
    data['thresh'] = thresh
    data['page_title'] = '{}, {}: Topic Pair Network I(a;b) >= {}'.format(slug, trial, thresh)
    data['nodes'], data['edges'] = els.get_topicpair_net(thresh)
    return render_template("topic_pair_net.html", **data)


@app.route('/projects/<slug>/<trial>/docs')
def doc_list(slug, trial='trial1'):
    """View doc list"""
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    set_project_menu(cfg, slug, trial)
    data['slug'] = slug
    data['trial'] = trial
    data['page_title'] = '{}, {}'.format(slug, trial)
    data['doc_entropy'] = els.get_doc_entropy()
    data['doc_entropy_avg'] = els.get_doc_entropy_avg()
    data['topic_entropy'] = data['doc_entropy_avg']
    data['docs'] = els.get_docs_for_topic_entropy(data['doc_entropy_avg'])
    return render_template('doc_list.html', **data)


@app.route('/projects/<slug>/<trial>/doc/<int:src_doc_id>')
def doc_item(slug, src_doc_id, trial='trial1'):
    """View doc item"""
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    set_project_menu(cfg, slug, trial)
    data['groups'] = cfg.get_group_fields()
    data['slug'] = slug
    data['trial'] = trial
    data['page_title'] = '{}, {} src_doc_id = {}'.format(slug, trial, src_doc_id)
    data['doc'] = els.get_doc(src_doc_id)
    data['doc_id'] = els.get_doc_id_for_src_doc_id(src_doc_id)
    data['doctopics'] = els.get_topics_for_doc_id(data['doc_id'])
    data['topics'] = els.get_topics()
    return render_template('doc_item.html', **data)


@app.route('/projects/<slug>/<trial>/docs/h/<topic_entropy>')
def docs_for_entropy(slug, topic_entropy, trial='trial1'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    set_project_menu(cfg, slug, trial)
    data['slug'] = slug
    data['trial'] = trial
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
    set_project_menu(cfg, slug, trial)
    data['slug'] = slug
    data['trial'] = trial
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
    data['group_label'] = cfg.ini['DEFAULT']['src_ord_col']
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


@app.route('/projects/<slug>/<trial>/groups/<group>')
def groups(slug, trial, group):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['slug'] = slug
    data['trial'] = trial
    data['group_field'] = group
    data['page_title'] = '{}, {}: Groups'.format(slug, trial)
    data['group_matrix'] = els.get_group_matrix(group)
    data['group_pairs'] = els.get_group_pairs(group)
    data['group_counts'] = els.get_group_counts(group)
    return render_template('group.html', **data)


@app.route('/projects/<slug>/<trial>/groups/<group>/<item>')
def group_item(slug, trial, group, item):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['slug'] = slug
    data['trial'] = trial
    data['group_field'] = group
    data['page_title'] = '{}, {}: {} = {}'.format(slug, trial, group, item)
    data['item'] = item
    data['topics'] = els.get_group_topics(group, item)
    data['comps'] = els.get_group_comps(group, item)
    data['docs'] = els.get_docs_for_group(item, group)
    data['max_tw'] = els.get_max_topic_weight()
    return render_template('group_item.html', **data)


@app.route("/projects/<slug>/<trial>/ngram/<ngram>")
def ngram_item(slug, trial, ngram):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['slug'] = slug
    data['trial'] = trial
    data['ngram'] = ngram
    data['degree'] = len(ngram.split('_'))
    data['page_title'] = '{}, {}: NGram: "{}"'.format(slug, trial, ngram.replace('_', ' '))
    data['docs'] = els.get_docs_for_ngram(ngram, data['degree'])
    data['groups'] = els.get_ngrams_per_group(ngram, data['degree'])
    return render_template('ngram_item.html', **data)


@app.route("/projects/<slug>/<trial>/pca")
def pca_page(slug, trial='trial1'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['slug'] = slug
    data['trial'] = trial
    data['page_title'] = 'PCA'
    data['pca_docs'] = els.get_pca_docs()
    data['pca_terms'] = els.get_pca_terms()
    data['pca_items'] = els.get_pca_items()
    data['max_variance'] = data['pca_items']['explained_variance'].max()
    data['topics'] = els.get_topics()
    return render_template('pca.html', **data)


@app.route("/projects/<slug>/<trial>/w2v")
@app.route("/projects/<slug>/<trial>/w2v/<join>")
def w2v_page(slug, trial='trial1', join='inner'):
    cfg = get_project_config(slug)
    els = Elements(cfg, trial)
    data['slug'] = slug
    data['trial'] = trial
    data['page_title'] = 'Word Embeddings'
    data['join'] = join
    data['topics']  = els.get_topics()
    data['coords'] = els.get_tsne_coords(join=join)
    return render_template('w2v.html', **data)
    

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


def set_project_menu(cfg, slug, trial):
    path_prefix =  '/projects/{}/{}'.format(slug, trial)
    data['sub_menu'] = [("{}".format(path_prefix), "Project")]
    for group_field in cfg.ini['DEFAULT']['group_fields'].split(','):
        group_field = group_field.strip()
        group_field_label = group_field.replace('doc_', '').replace('_', ' ')
        data['sub_menu'].append(("{}/topic_heatmap/{}".format(path_prefix, group_field),
                                    "Topic/{} Heatmap".format(group_field_label)))
    data['sub_menu'].append(("{}/topic_pair_net".format(path_prefix),
                             "Topic Pair Network"))
    # data['sub_menu'].append(("{}/topic_pair_heatmap/jsd".format(path_prefix),
    #                          "Topic Pair Similiarity Heatmap"))
    # data['sub_menu'].append(("{}/topic_pair_heatmap/i_ab".format(path_prefix),
    #                          "Topic Pair Contiguity Heatmap"))
    data['sub_menu'].append(("{}/docs".format(path_prefix), "Documents"))
    data['sub_menu'].append(("{}/pca".format(path_prefix), "Principle Components"))
    data['sub_menu'].append(("{}/w2v".format(path_prefix), "Word Embeddings"))


if __name__ == '__main__':
    app.run(debug=True)
