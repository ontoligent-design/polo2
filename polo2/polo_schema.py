from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

metadata = MetaData()

src_doc = Table('src_doc', metadata,
                    Column('doc_id', Integer, primary_key=True),
                    Column('doc_title', String),
                    Column('doc_date', String),
                    Column('doc_year', Integer),
                    Column('doc_label', String),
                    Column('doc_original', String),
                    Column('doc_content'),
                    Column('doc_url')
                )