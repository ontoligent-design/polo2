import sqlite3
import sys
import pandas as pd

"""
* Add schema information so that sqlite3 tables can have indexes made
* Consider writing classes for Schema and Table, the latter being subclassable with table building methods
"""

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
            self.tables[table_name] = df.reset_index()

    def get_table(self, table_name=''):
        if self.cache_mode and table_name in self.tables:
            return self.tables[table_name]
        else:
            cur = self.conn.cursor()
            cur.execute("select count(*) from sqlite_master where type='table' and name=?", (table_name,))
            sql_check = cur.fetchone()[0]
            if sql_check:
                sql = 'select * from {}'.format(table_name)
                df = pd.read_sql_query(sql, self.conn)
                if self.cache_mode:
                    self.tables[table_name] = df
                return df
                #    return self.tables[table_name]
                #else:
                #    return df
            else:
                sys.exit("Table `{}` needs to be created first.".format(table_name))

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