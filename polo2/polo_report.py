from polo2 import PoloDb
import pandas as pd

class PoloReport(PoloDb):

    def __init__(self, config, trial):
        if trial not in config.trials:
            raise ValueError("Invalid trail name `{}`.format(trial)")
        self.config = config
        self.trial = trial
        self.slug = self.config.ini['DEFAULT']['slug']
        self.base_path = self.config.ini['DEFAULT']['base_path']
        dbfile = "{}/{}-mallet-{}.db".format(self.base_path, self.slug, self.trial)
        PoloDb.__init__(self, dbfile)
        self.thresh = float(self.config.ini['DEFAULT']['thresh'])

    def get_topics_for_doc(self, doc_id):
        doctopics = self.get_table('doctopic')
        topics = self.get_table('topic')
        data = doctopics[(doctopics['topic_weight'] >= self.thresh) &
                  (doctopics['doc_id'] == doc_id)][['topic_id', 'topic_weight']].sort_values('topic_weight')
        labels = topics.loc[data.topic_id].apply(lambda x: '{} T-{}'.format(x.topic_words, x.topic_id), axis=1)
        return (data, labels)

    def show_topics_for_doc(self, doc_id):
        (data, labels) = self.get_topics_for_doc(doc_id)
        fig_title = 'Topics for Doc {}'.format(doc_id)
        ax = data.plot(x='topic_id', y='topic_weight', kind='barh', title=fig_title, legend=False)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Average Topic Weight')
        ax.set_ylabel('Topic')
        fig = ax.get_figure()
        return fig
