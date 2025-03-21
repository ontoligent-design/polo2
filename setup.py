from setuptools import setup, find_packages
from codecs import open
from os import path

# todo: Add a data path to store templates and boilerplate

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'polo2',
    version = '0.2', # Versions should comply with PEP440.
    description = 'Polo lets you play with Mallet and much more',
    long_description = long_description,
    url = 'https://github.com/ontoligent-design/polo2',
    author = 'Rafael C. Alvarado',
    author_email = 'ontoligent@gmail.com',
    license = 'MIT',
    keywords = 'natural language processing, text mining, digital humanities',
    packages = ['polo2'], # packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires = ['pandas', 'lxml', 'scipy', 'numpy', 'textblob', 'nltk'],
    scripts = ['bin/polo', 'bin/posh'],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[

        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    #extras_require={
    #    'dev': ['check-manifest'],
    #    'test': ['coverage'],
    #},

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #package_data={
    #    'templates': ['config.ini'],
    #},
    #include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #data_files=[('templates', ['config.ini'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    #entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
)
