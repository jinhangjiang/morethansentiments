from setuptools import setup

# with open("README.md", "r") as f:
#     long_description = f.read()



setup(
    name = 'MoreThanSentiments',
    version = '0.1.2',
    description = 'An NLP python package for computing Boilerplate score and many other text features.',
    py_modules = ["MoreThanSentiments"],
    package_dir = {'':'src'},
#     package = ['morethansentiments'],
    author = 'Jinhang Jiang, Karthik Srinivasan',
    author_email = 'jinhang@asu.edu',
    long_description = open('README.md').read() + '\n\n' + open('CHANGELOG.md').read(),
    long_description_content_type = "text/markdown",
    url='https://github.com/jinhangjiang/morethansentiments',
    include_package_data=True,
    
    
    classiffiers = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: BSD License",
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Text Processing',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
    ],
    
    install_requires = [
        'tqdm ~= 4.59.0',
        'spacy ~= 3.3.0',
        'pandas ~= 1.2.4',
        'nltk ~= 3.6.1',
    ],
    
    keywords = ['Text Mining', 'Data Science', 'Natural Language Processing', 'Accounting'],
    
)