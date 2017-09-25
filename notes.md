# Design Notes

Polo2 is meant to simplify Polo, which now how too many things stuck together. Polo2 should be focused on running mallet and putting the results in a sqlite file. The main value adds are:

* Simplify the running of `mallet` by using config files, one for polo itself and one or more for the projects that use polo. Also, polo will be a shell command, ideally in the path, that will run some basic arguments beyond those implied by project config files.
* Simplify the use of `mallet`'s results by putting them into a normalized relational database file using `SQLite`.

Requirements:

* `polo` tool that reads an `ini` file and runs `mallet` and generates a `sqlite` file for use in other applications.
* `polo` expects:
  * a polo config file
  * a project config file
  * mallet
  * a corpus in mallet format
  * sqlite3 (come with python)