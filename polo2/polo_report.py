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


    def get_topicpair_by_group(self, group_col):

        # Get the groups
        #groups =

        # Get doc count to calculate topic frequencies
        r = self.conn.execute("select count() from doc")
        doc_num = int(r.fetchone()[0])

        # Create the doctopic matrix dataframe
        doctopic = self.get_table('doctopic', set_index=True)
        dtm = doctopic.unstack()
        if dtm.columns.nlevels == 2:
            dtm.columns = dtm.columns.droplevel()
        del doctopic

        # Add topic frequency data to topic table
        topic = self.get_table('topic', set_index=True)
        topic['topic_freq'] = topic.apply(lambda x: len(dtm[dtm[x.name] > self.cfg_thresh]), axis=1)
        topic['topic_rel_freq'] = topic.apply(lambda x: x.topic_freq / doc_num, axis=1)
        self.put_table(topic, 'topic', index=True)

        # Create topicword matrix dataframe
        topicword = self.get_table('topicword', set_index=True)
        topicword['word_count'] = topicword['word_count'].astype(int)
        twm = topicword.unstack().fillna(0)
        if twm.columns.nlevels == 2:
            twm.columns = twm.columns.droplevel(0)
        del topicword

        # Create topicpair dataframe
        from itertools import combinations
        pairs = [pair for pair in combinations(topic.index, 2)]
        topicpair = pd.DataFrame(pairs, columns=['topic_a_id', 'topic_b_id'])

        # Calculate distances by document vector
        #topicpair['cosim_doc'] = topicpair.apply(lambda x: pm.cosine_sim(dtm[x.topic_a_id], dtm[x.topic_b_id]), axis=1)
        #topicpair['jscore_doc'] = topicpair.apply(lambda x: pm.jscore(dtm[x.topic_a_id], dtm[x.topic_b_id]), axis=1)
        #topicpair['jsd_doc'] = topicpair.apply(lambda x: pm.js_divergence(dtm[x.topic_a_id], dtm[x.topic_b_id]), axis=1)

        # Calculate distances by word vector -- SHOULD BE MORE ACCURATE
        topicpair['cosim'] = topicpair.apply(lambda x: pm.cosine_sim(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        topicpair['jscore'] = topicpair.apply(lambda x: pm.jscore(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)
        topicpair['jsd'] = topicpair.apply(lambda x: pm.js_divergence(twm[x.topic_a_id], twm[x.topic_b_id]), axis=1)

        # Calculate PWMI
        def get_p_ab(a, b):
            p_ab = len(dtm[(dtm[a] > self.cfg_thresh) & (dtm[b] > self.cfg_thresh)]) / doc_num
            return p_ab
        topicpair['p_ab'] = topicpair.apply(lambda x: get_p_ab(x.topic_a_id, x.topic_b_id), axis=1)
        topicpair['p_aGb'] = topicpair.apply(lambda x: x.p_ab / topic.loc[x.topic_b_id, 'topic_rel_freq'], axis=1)
        topicpair['p_bGa'] = topicpair.apply(lambda x: x.p_ab / topic.loc[x.topic_a_id, 'topic_rel_freq'], axis=1)
        def get_pwmi(a, b, p_ab):
            if p_ab == 0: p_ab = .000001  # To prevent craziness in prob calcs
            p_a = topic.loc[a, 'topic_rel_freq']
            p_b = topic.loc[b, 'topic_rel_freq']
            i_ab = pm.pwmi(p_a, p_b, p_ab)
            return i_ab
        topicpair['i_ab'] = topicpair.apply(lambda x: get_pwmi(x.topic_a_id, x.topic_b_id, x.p_ab), axis=1)
        topicpair['x_ab'] = topicpair.apply(lambda x: (x.p_aGb + x.p_bGa) / 2, axis=1)

        topicpair.set_index(['topic_a_id', 'topic_b_id'], inplace=True)
        self.put_table(topicpair, 'topicpair', index=True)
