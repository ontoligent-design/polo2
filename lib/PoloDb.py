import sqlite3
import sys
import pandas as pd

class PoloDb:

    # Use to store tables in memory
    tables = {}
    cache_mode = False

    def __init__(self, dbfile):
        self.dbfile = dbfile
        try:
            self.conn = sqlite3.connect(self.dbfile)
        except sqlite3.Error as e:
            print("Can't connect to database:", e.args[0])
            sys.exit(0)

    def __del__(self):
        try:
            self.conn.close()
        except sqlite3.Error as e:
            print("Can't close database:", e.args[0])

    def put_table(self, df, table_name='test', if_exists='replace', index=False, index_label=None):
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=index, index_label=index_label)
        if self.cache_mode:
            self.tables[table_name] = df

    def get_table(self, table_name=''):
        """Needs to be SQL safe and check if table exists!"""
        if self.cache_mode and table_name in self.tables[table_name]:
            return self.tables[tablename]
        else:
            sql = 'select * from {}'.format(table_name)
            df = pd.read_sql_query(sql, self.conn)
            if self.cache_mode:
                self.tables[table_name] = df
                return self.tables[table_name]
            else:
                return df

    def clear_table_cache(self):
        for table_name in self.tables:
            self.tables.pop(table_name, None)