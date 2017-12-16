import sqlite3, re
import pandas as pd


# todo: Integrate SQLAlchemy to add indexes, etc.

class PoloDb():

    tables = {} # Used to cache tables
    cache_mode = False

    def __init__(self, dbfile, read_only=False):
        self.dbfile = dbfile
        self.read_only = read_only
        try:
            self.conn = sqlite3.connect(self.dbfile)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise ValueError("Can't connect to database:", e.args[0])

    def __del__(self):
        if hasattr(self, 'conn'):
            try:
                self.conn.close()
            except sqlite3.Error as e:
                raise ValueError("Can't close database:", e.args[0])

    def put_table(self, df, table_name='test', if_exists='replace', index=False, index_label=None):
        if not self.read_only:
            df.to_sql(table_name, self.conn, if_exists=if_exists, index=index, index_label=index_label)

            if self.cache_mode:
                self.tables[table_name] = df.reset_index() # Index reset is crucial
        else:
            # fixme: Change ValueErrors to proper errors
            raise ValueError('Read-only mode for safety.')

    def get_table(self, table_name = '', set_index = False):
        if self.cache_mode and table_name in self.tables:
            df = self.tables[table_name]
            if set_index:
                df = self._set_index(df) # fixme: Find index fields to set
            return df
        else:
            cur = self.conn.cursor()
            cur.execute("select count(*) from sqlite_master where type='table' and name=?", (table_name,))
            sql_check = cur.fetchone()[0]
            if sql_check:
                sql = 'select * from {}'.format(table_name)
                df = pd.read_sql_query(sql, self.conn)
                if set_index:
                    df = self._set_index(df)
                if self.cache_mode:
                    self.tables[table_name] = df
                return df
            else:
                raise ValueError("Table `{}` needs to be created first.".format(table_name))

    def _set_index(self, df):
        idx = [col for col in df.columns if re.search(r'_id$', col)]
        if len(idx):
            df.set_index(idx, inplace=True)
        else:
            raise ValueError('No index field to set.')
        return df

    # todo: Finish writing the method add_pkeys_to_tables()
    def  add_pkeys_to_tables(self):
        """Add primary keys to db tables"""
        res1 = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for table in [row[0] for row in res1.fetchall()]:
            res2 = self.conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,))
            info = res2.fetchone()
            if (re.search(r'PRIMARY KEY', info[0])):
                continue
            id_cols = [re.sub(r'\W+', '', token) for token in info[0].split() if re.search('_id', token)]
            alter_sql = ''

    def get_table_names(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        tdfs = []
        for table in tables:
            tdfs.append(pd.read_sql_query("select '{0}' as table_name, count(*) as nrows from {0}".format(table[1]), self.conn))
        tables_df = pd.concat(tdfs, axis=0)
        tables_df.set_index('table_name', inplace=True)
        return(tables_df)

    def clear_table_cache(self):
        for table_name in self.tables:
            self.tables.pop(table_name, None)

