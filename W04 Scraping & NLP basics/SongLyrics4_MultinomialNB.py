# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:00:38 2019
@author: alexl
"""

import re
import spacy

import requests
import pandas as pd
from bs4 import BeautifulSoup as soup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


regex = 'href=\"([^\"]+)\"'
spacy_encode = spacy.load('en_core_web_md')  # Medium database
#spacy_encode = spacy.load('en_core_web_sm')  # Small database
artists_names = []

def ArtistList():
    """
    Returns a list of names from user input.
    CURRENTLY NOT BEING USED. STR can't have attributes.
    """
    names = []
    names.append(input("Please enter an artist name: "))
    while names[-1] != "none":
        names.append(input("Please enter an artist name or none: ")) #.replace(" ","_"))
    
    return names[:-1]


class ArtistProfile():
    
    def __init__(self, name):
        """
        Creates variables for the object
        """
        self.name = name.replace(" ", "-")
        self.list_urls = [f'https://www.metrolyrics.com/{self.name}-alpage-1.html']
        self.song_urls = []
        self.song_lyrics = []
        self.token_lyrics = []
        
        # Sets up the inital page to allow GetListUrls and GetSongUrls to function
        self.request = requests.get(self.list_urls[0])
        self.request = soup(self.request.text, 'html.parser')
        self.request = self.request.find_all(attrs = {"class":"switchable lyrics clearfix"})

    
    def GetListUrls(self):
        """
        Extracts the additional page URLs from
        the initial page's pagination field.
        """
        try:
            self.pagination = self.request[0].find_all(attrs = {"class":"pagination"})
            self.pagination = self.pagination[0].find_all(attrs = {"class":"pages"})
            for link in self.pagination[0].find_all('a'):
                self.list_urls.append(link.get('href'))
            
            # Ensure no duplicated pages
            self.list_urls = list(set(self.list_urls))
        except:
            # Ensure no duplicated pages
            self.list_urls = list(set(self.list_urls))

        
    def GetSongUrls(self, url):
        """
        Gets the lyrics from each URL and clean in into a list
        """
        self.request = requests.get(url)
        self.request = soup(self.request.text, 'html.parser')
        self.request = self.request.find_all(attrs = {"class":"switchable lyrics clearfix"})
        
        # Extracts urls for song lyrics from the current page's "popular" list
        self.pop_list = self.request[0].find_all(attrs = {"class":"module", "id":"popular"})
        for link in self.pop_list[0].find_all('a'):
            self.song_urls.append(link.get('href'))
            
        print(f"Found {len(self.song_urls)} songs by {self.name}")

            
    def GetLyrics(self):
        """
        Extracts lyrics from a song URL
        """
        for song in self.song_urls:
            song_request = requests.get(f'{song}')
            song_request = soup(song_request.text, 'html.parser')
            song_request = song_request.find_all(attrs = {"class":"verse"})
            song_request = re.sub("<.+?>", '', str(song_request)) #takes out HTML code
            song_request = re.sub("\s+", " ", str(song_request)) #removes new lines
            self.song_lyrics.append(song_request[1:-1]) #removes square brackets
        
        # Ensure no duplicated lyrics and remove all but 1 na
        self.song_lyrics = list(set(self.song_lyrics))[1:]
        print(f"Extracted {len(self.song_lyrics)} song lyrics from {self.name}")


    def ClassBalancer(self, length):
        """
        Creates a "song_df" that is equal
        in length to the other artist(s)
        """
        self.song_df = pd.DataFrame(self.token_lyrics[:length])
        self.song_df['name'] = self.name
        
        
    def SpacyEncoder(self):
        """
        Transforms song lyrics into spacy encoded words.
        Then cleans out punctuation and stop words.
        """
        for song in self.song_lyrics:
            token_song = spacy_encode(song)
            
            cleaned_lyrics = ''
            for word in token_song:
                    if not word.is_stop and word.lemma_ != '-PRON-' and word.pos_ != 'PUNCT' and word.is_alpha == True:
                        cleaned_lyrics += word.text + ' '

            self.token_lyrics.append(cleaned_lyrics)
        print(f"Spacy has encoded {len(self.token_lyrics)} songs by {self.name} into token_lyrics")
        
        
    def ExportCSV(self):
        """
        Exports Lyrics into CSV file
        """
        lyrics_df = pd.DataFrame(data = self.song_lyrics)
        lyrics_df.dropna(inplace = True)
        lyrics_df.to_csv(f'Data/song_lyrics_{self.name}.csv')


    def __repr__(self):
        
        return f"this is a return for debugging"
    
    

def Vectorizer(artist1, artist2):
    """
    Takes in the artists lyrics and creates a single
    dataframe which then gets vectorized.
    """
    df = pd.concat([artist1.song_df, artist2.song_df], axis=0, join='inner', ignore_index=True)

    y = df['name']
#    y, keys = pd.factorize(y)


    df = df.drop('name',axis = 1)
    # Initialize the Vectorizer
    tf = TfidfVectorizer()
    
    # Fit it to the single df
    tf.fit(df[0])
    vecdf = tf.transform(df[0])

    return vecdf, y, tf
    

def NaiveBayesNB(X, y):
    
    m = MultinomialNB()

    m.fit(X, y)
    print(f"MultinomialNB score is: {m.score(X, y)}")
    
    return m


def GuessWho(tf, m):
    """
    Loops to ask user input and guesses who of the two artists would be
    most likely to sing it. Until "stop" is called.
    """
    user_input = ''
    while user_input != 'stop':
        user_string = []
        user_input = (input('Please enter any series of words to see who would be more likely to sing them: '))
        user_string.append(user_input)
        user_string_transformed = tf.transform(user_string)
        answer = m.predict(user_string_transformed)
        proba = m.predict_proba(user_string_transformed)
        print(answer, proba)


def ScraperWorkflow(self):
    """
    Scrapes song lyrics for an artist and cleans it into a list.
    """
    # Goes to initial page and grabs all the pagination links
    self.GetListUrls()
    
    # Extracts the urls for each indvidual song
    for list_page in self.list_urls:
        self.GetSongUrls(list_page)
    
    # Extracts the lyrics for each song
    self.GetLyrics()
    
    self.SpacyEncoder()
    # Exports lyrics to csv
#    artist.ExportCSV()


def VectorWorkflow(artist1, artist2):
    """
    Makes each artist's lyrics list the same size.
    Concats them together with "name" column and vectorizes them.
    
    """
    shortest_dataset = min(len(artist1.song_lyrics), len(artist2.song_lyrics))
    artist1.ClassBalancer(shortest_dataset)
    artist2.ClassBalancer(shortest_dataset)
    
    # Vectorizes lyrics using Bag of Words
    X, y, tf = Vectorizer(artist1, artist2)
    
    # train the NaiveBayes model
    m = NaiveBayesNB(X, y)

    GuessWho(tf, m)



# Initiates the artist
artist1 = ArtistProfile('virzha')
ScraperWorkflow(artist1)

artist2 = ArtistProfile('zhavia')
ScraperWorkflow(artist2)


VectorWorkflow(artist1, artist2)













