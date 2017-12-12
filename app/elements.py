from polo2 import PoloDb
import pandas as pd

class Elements(object):

    def __init__(self, config, trial_name='trial1'):

        # Set some values
        if trial_name not in config.trials:
            raise ValueError("Invalid trail name `{}`.format(trial)")
        self.config = config
        self.trial = trial_name
        self.slug = self.config.ini['DEFAULT']['slug']
        self.base_path = self.config.ini['DEFAULT']['base_path']
        self.thresh = float(self.config.ini['DEFAULT']['thresh'])

        # Load the databases
        corpus_db_file = self.config.generate_corpus_db_file_path()
        model_db_file = self.config.generate_model_db_file_path(self.trial)
        self.corpus = PoloDb(corpus_db_file)
        self.model = PoloDb(model_db_file)

    def get_doc_count(self):
        self.doc_count = pd.read_sql_query('SELECT count(*) AS n FROM doc', self.corpus.conn).n.tolist()[0]
        return self.doc_count

    def get_topic_count(self):
        self.topic_count = pd.read_sql_query('SELECT count(*) AS n FROM topic', self.model.conn).n.tolist()[0]
        return self.topic_count

    def get_topic_list(self, by_alpha = True):
        topics = self.model.get_table('topic', set_index=True)
        alpha_max = topics.topic_alpha.max()

        from scipy import stats
        topics['topic_alpha_zscore'] = stats.zscore(topics.topic_alpha)
        topics['topic_alpha_percent'] = ((topics.topic_alpha / alpha_max) * 100).astype(int)
        topics['topic_alpha_zsign'] = topics.topic_alpha_zscore.apply(lambda x: 'pos' if x > 0 else 'neg')

        num_topics = len(topics.index)
        sql = "SELECT topic_id, GROUP_CONCAT(topic_phrase, ', ') as phrases FROM topicphrase " \
              "GROUP BY topic_id ORDER BY phrase_weight DESC"
        phrases = pd.read_sql_query(sql, self.model.conn)
        phrases.set_index('topic_id', inplace=True)
        cards = []
        if by_alpha:
            topic_id_list = topics.topic_alpha.sort_values(ascending=False).index.tolist()
        else:
            topic_id_list = range(num_topics)
        for topic_id in topic_id_list:
            card = dict(
                topic_id = topic_id,
                topic_alpha = round(topics.loc[topic_id].topic_alpha, 5),
                topic_alpha_zscore = round(topics.loc[topic_id].topic_alpha_zscore, 5),
                topic_alpha_zsign = topics.loc[topic_id].topic_alpha_zsign,
                topic_alpha_percent = topics.loc[topic_id].topic_alpha_percent,
                topic_phrases = phrases.loc[topic_id].phrases,
                topic_words = topics.loc[topic_id].topic_words
            )
            cards.append(card)
        return cards

    def display_topic_list(self, by_alpha=True, format='simple_table'):
        topic_list = self.get_topic_list(by_alpha)
        if format == 'simple_table':
            df = pd.DataFrame(topic_list)
            el = df.to_html()
        elif format == 'w3_cards':
            cards = []
            for topic in topic_list:
                alpha_bar = self.make_progress_bar(color='blue', percent=topic['topic_alpha_percent'])
                card = """
                <tr>
                    <td>
                        <div class="topic_id">{}</div>
                        <div class="topic_alpha">{}</div>
                        <div class="topic_alpha_zscore {}">{}</div>
                    </td>
                    <td>
                        {}
                        <div class="topic_words">{}</div>
                        <div class="topic_phrases">{}</td>
                    </td>
                </tr>
                """.format(topic['topic_id'], topic['topic_alpha'], topic['topic_alpha_zsign'], topic['topic_alpha_zscore'], alpha_bar, topic['topic_words'], topic['topic_phrases'])
                cards.append(card)
            el = ("<table>{}</table>").format('\n'.join(cards))
        else:
            el = "NO FORMAT GIVEN"
        return el

    def get_prhases_for_topic(self, topic_id):
        sql = "SELECT topic_phrase FROM topicphrase WHERE topic_id = {} ORDER BY phrase_weight DESC".format(topic_id)
        phrases = ', '.join(pd.read_sql_query(sql, self.model.conn).topic_phrase.tolist())
        return phrases

    def make_progress_bar(self, color='grey', height=24, percent = 0, content=''):
        bar = """<div class="w3-light-{0}">
        <div class="w3-{0}" style="height:{1}px; width:{2}%">{3}</div>
        </div>""".format(color, height, percent, content)

        return bar


