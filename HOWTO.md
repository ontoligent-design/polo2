# How to use MALLET on your own corpus

## Synopsis

## Steps to generate the corpus database

### Scrape some data and put it into a subdirectory of /raw

The first thing to do is get some corpus data. This is usually done by scraping pages or posts from
a web site and saving them in a directory. You may also save them directory to a database. In any case,
try to preserve as much of the metadata as possible, such as the title, date, and unique ID associated
with each docunent. The ID should be something that you can append to a base URL to get to the original document.

### Create an entry in `config.ini` for the project

Next, create an entry in the `config.ini` file. It should look something like this:

```
[reddit]    
name                = 'Reddit'
raw_files           = ''
db_file             = ./db/reddit.db
num_topics          = 200
num_iterations      = 1000
mallet_corpus_input = ./models/input/reddit-mallet-corpus.csv
extra_stops         = ./models/input/extra-stopwords.csv
``` 

The string in the square brackets is key -- it will be used as the ID
for the project later. Note that `num_topics` and `num_iterations`,
which are parameters sent to `mallet`, can be overridden at runtime.

### Create a class in `sources.py` for the source data that implements the abstract method `import_src_files()`

Now edit `sources.py` and create a class that corresponds to the entry
you made in `config.ini`. If you have a project named `reddit` you
need to make a class in `sources.py` called `Reddit`. This class
should have one method -- `import_src_files()`. This method does one
thing: it grabs the data you scroped and puts it into the database
table called `src_data`. So, basically you open your data source,
whehter it's a directory with files or a database table, and loop
through it, grabbing the data in each loop and inserting it into the
target database.

Here is an example for the Reddit class:

```python

class Reddit(Source):

    def import_src_files(self):

        # Change this to pull in the data from the db

        #CREATE TABLE works (ID,TITLE,BODY,ATTACH,COMMENTS);

        #files = glob.glob('{}/*.json'.format(self.src_dir))
        with sqlite3.connect('raw/reddit/reddit.db') as reddit, self.dbi as db:

            cur = reddit.cursor()
                                
            meta = OrderedDict()
            meta['src_meta_id'] = 'reddit'
            meta['src_meta_desc'] = 'AskHitorians'
            meta['src_meta_base_url'] = 'https://www.reddit.com/r/AskHistorians'
            db.conn.execute('INSERT INTO src_doc_meta (src_meta_id,src_meta_desc,src_meta_base_url) VALUES (?,?,?)',tuple(meta.values()))
            db.conn.commit()

            self.get_max_doc_id()
            doc_id = self.max_doc_id

            re_year = re.compile(r'^(\d{4})')

            # Add posts
            self.get_max_doc_id()
            doc_id = self.max_doc_id
            sql = "SELECT post_id, date, title, post_author, body FROM posts"
            rs = db.conn.execute(sql)
            for r in rs:
                values = OrderedDict()
                doc_id += 1

                values['doc_id']        = doc_id
                values['doc_uri']       = meta['src_meta_base_url'] + '/{}/'.format(r[0])
                values['doc_title']     = r[2]
                values['doc_original']  = r[4]
                values['doc_content']   = self.clean_text(r[2] + ' ' + r[4])
                values['doc_date']      = r[1]
                m = re_year.match(r[1])
                year = m.group(1)
                values['doc_year']      = year
                values['doc_label']     = year
                if len(r[4]) >= 140:
                    self.insert_row('src_doc',values)

            # Add comments
            sql = "SELECT c.post_id, c.date, c.comment_id, c.comment_author, c.comment, p.title FROM comments c JOIN posts p USING (post_id)"
            rs = db.conn.execute(sql)
            for r in rs:
                values = OrderedDict()
                doc_id += 1

                values['doc_id']        = doc_id
                values['doc_uri']       = meta['src_meta_base_url'] + '/comments/{}/'.format(r[2])
                values['doc_title']     = 'Comment {} to: {}'.format(r[2],r[5])
                values['doc_original']  = r[4]
                values['doc_content']   = self.clean_text(r[4])
                values['doc_date']      = r[1]
                m = re_year.match(r[1])
                year = m.group(1)
                values['doc_year']      = year
                values['doc_label']     = year
                if len(r[4]) >= 140:
                    self.insert_row('src_doc',values)

```

The method itself needs to update the `src_doc_meta` table as well as the `src_doc` table.

Classes will vary in design because document sources vary. You may need to import other libraries
to make your import work.

### Add an entry to the `SourceFactory` factory method

Add an entry to the factory method for `SourceFactory` in `sources.py` like this:

```python
    if type == "reddit": return Reddit(db_file,raw_files)
```

### Run `make.py`

Run `make.py` with your project name and run level as arguments. Run levels refer to the specific 
actions you want to perform on your data. To see avaiable run levels, just type `./make.py` at 
the command line. Run levels are designed to allow you to intervene at each step in the pipeline
and regenerate everything that depends on what your re-run. For example, if you change your 
source data, you will want to rerun the topic model. You can also run each function independently.

### Copy or link the resulting database to /home/shared/databases

## Steps to publish the database online (using the Topic Model Browser)

### Create an entry for the CodeIgniter database

This must be done by hand. Go to `http://198.211.100.224/phpmyadmin/` login using credentials available from the site admin.

In the table `db` in the database `mitre`, insert a new row with the following values:

* `db_name`: The short project name you created above
* `db_name`: A descriptive name for the database you created above
* `db_path`: The path to the database you created above

Note: All of this will be folded into a method later that you can invoke after creating the database.

### Create models, controllers, and views to interact with your data. 

Right now, there is the default [Topic Model Browser](http://198.211.100.224/mitre). 


