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
        #fig = ax.get_figure()
        #return fig

    def show_topic_index_table(self):
        topic_index = []
        topic = self.get_table('topic', set_index=True)
        topicphrase = self.get_table('topicphrase', set_index=True)
        for t in topic.sort_values('topic_alpha', ascending=False).index:
            tlab = '<b>T{}</b>'.format(t)
            words = '<b>{}</b>'.format(topic.loc[t].topic_words)
            phrases = ', '.join(topicphrase.loc[t].sort_values('phrase_weight', ascending=False).topic_phrase.tolist())
            scale = '<span style="font-size:8pt;color:lightblue;margin:0;padding:0;">{}</span>'.format(
                '&#9608;' * int(round(topic.loc[t].topic_alpha * 100)))
            alpha = '<tt>{:6f}</tt>'.format(topic.loc[t].topic_alpha)
            topic_index.append('<tr><td style="vertical-align:top;">{}</td></tr>'.format(
                '</td><td style="text-align:left;vertical-align:top;">'.join(
                    [tlab, alpha + ' ' + scale + '<br>' + words + '<br>' + phrases])))
        topic_table = '<table style="font-size:12pt;">{}</table>'.format(''.join(topic_index))
        return topic_table