"""
Note: This file may adapt (where noted) some code from the fnc-1-baseline
feature_engineering.py code.
"""

from gavin_utils.prep_data import MyDataSet

import re
import nltk
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer
)
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from tqdm import tqdm


class FeatureGeneration(object):

    save_name = '{name}_featurized.pkl'

    text_cols = ['Headline', 'body']
    re_whitespace = re.compile(r'\W+')
    re_punct = re.compile(r'[^a-zA-Z\d\s]')
    stopwords = nltk.corpus.stopwords.words('english')
    _wnl = nltk.WordNetLemmatizer()
    _n_gram_range = (1, 3)  # NOTE: Optimize?
    # NOTE: This is from the baseline feature engineering script.
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def __init__(self, df, name='train'):
        """Takes a df with 'body' and 'headline' columns"""
        self.attempt_reload(df, name)  # attempt to reload the df given name

    def attempt_reload(self, df, name):
        """If cleaning is done, save time and reload saved. Otherwise, redo and save"""
        try:
            self.df = pd.read_pickle(self.save_name.format(name=name))
            print("data reloaded")
        except FileNotFoundError:
            self.df = df
            # Cleaning text operations
            self.clean_text()
            # Adding features
            self._save(name)

    def _save(self, name):
        print("Saving...")
        self.df.to_pickle(self.save_name.format(name=name))
        print("Done.")

    def clean_text(self):
        print("Modifying df inplace... (may take a while)")
        self.clean_case_non_an()
        print('.', end='')
        self.lemmatize()
        print('.', end='')
        self.drop_stopwords()
        print('.', end='')
        print('Done.')

    def clean_case_non_an(self):
        """Coerce everything to lowercase and remove non-alphanumeric chars"""
        for col in self.text_cols:
            self.df[col] = (
                self.df[col]
                .str.lower()
                .str.replace(self.re_whitespace, ' ')
                .str.replace(self.re_punct, '')
            )

    def lemmatize(self):
        """Convert text columns to list of words"""
        for col in self.text_cols:
            self.df[col] = self.df[col].apply(
                lambda x: [self._wnl.lemmatize(w) for w in nltk.word_tokenize(x)]
            )

    def drop_stopwords(self):
        """Drop stopwords for column lists"""
        for col in self.text_cols:
            self.df[col] = self.df[col].apply(
                lambda x: [w for w in x if w not in self.stopwords]
            )

    def add_word_intersection(self):
        """
        Adds a new feature column of a set of words occurring in both headlines
        and body.

        This was partly inspired by one of the features in the baseline file though
        it may be duplicated by finding the cosine similarity between the headline
        and body count vectors.
        """
        # The set of words occurring in body
        set_body = self.df['body'].apply(lambda x: set(x))
        set_headline = self.df['Headline'].apply(lambda x: set(x))
        self.df['intersection'] = [x.intersection(y) for x, y in zip(set_body, set_headline)]
        intersection_count = [len(x) for x in self.df['intersection']]
        union = [x.union(y) for x, y in zip(set_body, set_headline)]
        union_count = [len(x) for x in union]
        self.df['intersection_pct'] = [
            inter / union for inter, union in zip(intersection_count, union_count)
        ]

    @property
    def all_text(self):
        """Create a body of text all text without duplicate docs"""
        text = list(self.df['Headline'].str.join(' ').unique())
        text.extend(list(self.df['body'].str.join(' ').unique()))
        return text

    def _get_cosine_similarities(self):
        headline_c_vector = self.c_vectorizer.transform(self.df['Headline'].str.join(' '))
        body_c_vector = self.c_vectorizer.transform(self.df['body'].str.join(' '))
        cosine_similarities = np.zeros(headline_c_vector.shape[0])

        for i, (headline_vect, body_vect) in enumerate(zip(headline_c_vector, body_c_vector)):
            cosine_similarities[i] = cosine_similarity(headline_vect, body_vect)
        return cosine_similarities

    def add_count_features(self):
        """
        Adds column "cosine_similarity_count" which is the cosine similarity measured
        between count vectors of headlines and bodies
        """
        # Ignore tokens that occur in more than 80% of docs or less than 2 times total
        self.c_vectorizer = CountVectorizer(
            ngram_range=self._n_gram_range, max_df=0.8, min_df=2
        )
        self.c_vectorizer.fit(raw_documents=self.all_text)
        self.df['cosine_similarity_count'] = self._get_cosine_similarities()

    def add_tfidf_features(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self._n_gram_range, max_df=0.8, min_df=2
        )
        self.tfidf_vectorizer.fit(raw_documents=self.all_text)
        # Create new variables to hold tf-idf vectors
        self.tfidf_headlines = self.tfidf_vectorizer.transform(self.df['Headline'].str.join(' '))
        self.tfidf_bodies = self.tfidf_vectorizer.transform(self.df['body'].str.join(' '))

        self.df['headline_tfidf_vec'] = self.tfidf_headlines
        self.df['body_tfidf_vec'] = self.tfidf_bodies
        cosine_similarities = np.zeros(self.df.shape[0])

        for i, (headline_vect, body_vect) in enumerate(zip(
            self.tfidf_headlines, self.tfidf_bodies
        )):
            cosine_similarities[i] = cosine_similarity(headline_vect, body_vect)
        self.df['cosine_similarity_tfidf'] = cosine_similarities
