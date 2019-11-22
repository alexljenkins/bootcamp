# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:00:38 2019
@author: alexl
"""

import re
import tqdm  # gives you status bars for loops/functions
import spacy
import requests
import pandas as pd
from bs4 import BeautifulSoup as soup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


SPACY_ENCODE = spacy.load('en_core_web_md')  # Medium database
# spacy_encode = spacy.load('en_core_web_sm')  # Small database
# REGEX = 'href=\"([^\"]+)\"'
# ARTISTS_NAMES = []

print("Initialised")  # TEXTING PRINT STATEMENT


class ArtistProfile():
    """
    Creates a new artist and finds their song lyrics
    """

    def __init__(self, name):
        """
        Creates variables for the object
        """
        self.name = name.replace(" ", "-")
        self.list_urls = [f'https://www.metrolyrics.com/{self.name}-alpage-1.html']
        self.song_urls = []
        self.song_lyrics = []
        self.token_lyrics = []
        self.pagination = []
        self.pop_list = []
        self.song_df = pd.DataFrame()

        # Sets up the inital page to allow get_list_urls and get_song_urls to function
        self.request = requests.get(self.list_urls[0])
        self.request = soup(self.request.text, 'html.parser')
        self.request = self.request.find_all(attrs={"class": "switchable lyrics clearfix"})

    def get_list_urls(self):
        """
        Extracts the additional page URLs from
        the initial page's pagination field.
        """
        try:
            self.pagination = self.request[0].find_all(attrs={"class": "pagination"})
            self.pagination = self.pagination[0].find_all(attrs={"class": "pages"})
            for link in self.pagination[0].find_all('a'):
                self.list_urls.append(link.get('href'))

            # Ensure no duplicated pages
            self.list_urls = list(set(self.list_urls))
            # TEXTING PRINT STATEMENT
            print(f"{self.name} has {len(self.list_urls)} pages of song lyrics")
        except:
            # Ensure no duplicated pages
            self.list_urls = list(set(self.list_urls))
            # TEXTING PRINT STATEMENT
            print(f"{self.name} only has {len(self.list_urls)} page of song lyrics")

    def get_song_urls(self, url):
        """
        Gets the lyrics from each URL and clean in into a list
        """
        self.request = requests.get(url)
        self.request = soup(self.request.text, 'html.parser')
        self.request = self.request.find_all(attrs={"class": "switchable lyrics clearfix"})

        # Extracts urls for song lyrics from the current page's "popular" list
        self.pop_list = self.request[0].find_all(attrs={"class": "module", "id": "popular"})
        for link in self.pop_list[0].find_all('a'):
            self.song_urls.append(link.get('href'))

        print(f"Found {len(self.song_urls)} songs by {self.name}")

    def get_lyrics(self):
        """
        Extracts lyrics from a song URL
        """
        print(f"Extracting song lyrics from {self.name}...")
        for song in self.song_urls:
            song_request = requests.get(f'{song}')
            song_request = soup(song_request.text, 'html.parser')
            song_request = song_request.find_all(attrs={"class": "verse"})
            song_request = re.sub("<.+?>", '', str(song_request))  # takes out HTML code
            song_request = re.sub("\s+", " ", str(song_request))  # removes new lines
            self.song_lyrics.append(song_request[1:-1])  # removes square brackets

        # Ensure no duplicated lyrics and remove all but 1 na
        self.song_lyrics = list(set(self.song_lyrics))[1:]
        print(f"Extracted {len(self.song_lyrics)} song lyrics from {self.name}")

    def class_balancer(self, length):
        """
        Creates a "song_df" that is equal
        in length to the other artist(s)
        """
        self.song_df = pd.DataFrame(self.token_lyrics[:length])
        self.song_df['name'] = self.name

    def spacy_encoder(self):
        """
        Transforms song lyrics into spacy encoded words.
        Then cleans out punctuation and stop words.
        """
        for song in self.song_lyrics:
            token_song = SPACY_ENCODE(song)

            cleaned_lyrics = ''
            for word in token_song:
                if not word.is_stop and word.lemma_ != '-PRON-' and word.pos_ != 'PUNCT' and word.is_alpha == True:
                    cleaned_lyrics += word.text + ' '

            self.token_lyrics.append(cleaned_lyrics)
        print(f"Spacy has encoded {len(self.token_lyrics)} songs by {self.name} into token_lyrics")

    def export_csv(self):
        """
        Exports Lyrics into CSV file
        """
        lyrics_df = pd.DataFrame(data=self.song_lyrics)
        lyrics_df.dropna(inplace=True)
        lyrics_df.to_csv(f'Data/song_lyrics_{self.name}.csv')

    def __repr__(self):
        return f"this is a return for debugging"


def vectorizer(artist1, artist2):
    """
    Takes in the artists lyrics and creates a single
    dataframe which then gets vectorized.
    """
    df = pd.concat([artist1.song_df, artist2.song_df], axis=0, join='inner', ignore_index=True)

    y = df['name']

    df = df.drop('name', axis=1)

    # Initialize the Vectorizer
    tf = TfidfVectorizer()

    # Fit it to the single df
    tf.fit(df[0])
    vecdf = tf.transform(df[0])

    return vecdf, y, tf


def naive_bayes_nb(X, y):
    """
    Fits a MultinominalNiaveBayes model to X and y.
    Don't need to factorize y for this particular model.
    """
    m = MultinomialNB()

    m.fit(X, y)
    print(f"MultinomialNB score is: {m.score(X, y)}")

    return m


def guess_who(tf, m):
    """
    Loops to ask user input and guesses who of the two artists would be
    most likely to sing it. Until "stop" is called.
    """
    user_input = ''
    while user_input != 'stop':
        user_string = []
        user_input = (
            input('Please enter any series of words to see who would be more likely to sing them: '))
        user_string.append(user_input)
        user_string_transformed = tf.transform(user_string)
        answer = m.predict(user_string_transformed)
        proba = m.predict_proba(user_string_transformed)
        print(answer, proba)


def scraper_workflow(self):
    """
    Scrapes song lyrics for an artist and cleans it into a list.
    """
    # Goes to initial page and grabs all the pagination links
    self.get_list_urls()

    # Extracts the urls for each indvidual song
    for list_page in self.list_urls:
        self.get_song_urls(list_page)

    # Extracts the lyrics for each song
    self.get_lyrics()

    self.spacy_encoder()
    # Exports lyrics to csv
#    artist.export_csv()


def vector_workflow(artist1, artist2):
    """
    Makes each artist's lyrics list the same size.
    Concats them together with "name" column and vectorizes them.

    """
    shortest_dataset = min(len(artist1.song_lyrics), len(artist2.song_lyrics))
    artist1.class_balancer(shortest_dataset)
    artist2.class_balancer(shortest_dataset)

    # Vectorizes lyrics using Bag of Words
    X, y, tf = vectorizer(artist1, artist2)

    # train the NaiveBayes model
    m = naive_bayes_nb(X, y)

    guess_who(tf, m)


# Initiates the artists
artist1 = ArtistProfile(str(input("Please enter an artist's name: ")))
artist2 = ArtistProfile(str(input("Please enter another artist's name: ")))

# artist1 = ArtistProfile('virzha')
scraper_workflow(artist1)

# artist2 = ArtistProfile('zhavia')
scraper_workflow(artist2)

vector_workflow(artist1, artist2)
