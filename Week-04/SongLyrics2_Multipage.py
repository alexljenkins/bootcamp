# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:00:38 2019
@author: alexl
"""

import requests
from bs4 import BeautifulSoup as soup
import re
import pandas as pd


regex = 'href=\"([^\"]+)\"'


class ArtistProfile():
    
    def __init__(self, name):
        """
        Creates variables for the object
        """
        self.name = name.replace(" ", "-")
        self.list_urls = [f'https://www.metrolyrics.com/{self.name}-alpage-1.html']
        self.song_urls = []
        self.song_lyrics = []
        
        # Sets up the inital page to allow GetListUrls and GetSongUrls to function
        self.request = requests.get(self.list_urls[0])
        self.request = soup(self.request.text, 'html.parser')
        self.request = self.request.find_all(attrs = {"class":"switchable lyrics clearfix"})

    
    def GetListUrls(self):
        """
        Extracts the additional page URLs from the initial page's pagination field.
        """
        self.pagination = self.request[0].find_all(attrs = {"class":"pagination"})
        self.pagination = self.pagination[0].find_all(attrs = {"class":"pages"})
        for link in self.pagination[0].find_all('a'):
            self.list_urls.append(link.get('href'))
        
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
            
        print(f"We found {len(self.song_urls)} number of songs by {self.name}")

            
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
        self.song_lyrics = list(set(self.song_lyrics))
        print(f"We've extracted {len(self.song_lyrics)} number of lyrics from {self.name}")
    
        
    def ExportCSV(self):
        """
        Exports Lyrics into CSV file
        """
        lyrics_df = pd.DataFrame(data = self.song_lyrics)
        lyrics_df.dropna(inplace = True)
        lyrics_df.to_csv(f'Data/song_lyrics_{self.name}.csv')


    def __repr__(self):
        
        return f"this is a return for debugging"


def Workflow():
    """
    Runs the entire application
    """
    # Sets the first artist
    rihanna = ArtistProfile("rihanna")
    
    # Goes to initial page and grabs all the pagination links
    rihanna.GetListUrls()
    
    # Extracts the urls for each indvidual song
    for list_page in rihanna.list_urls:
        rihanna.GetSongUrls(list_page)
    
    # Extracts the lyrics for each song
    rihanna.GetLyrics()
    
    # Exports it into a csv
    rihanna.ExportCSV()

Workflow()








# just in case the function above doesn't work this was the original
#ArtistProfile(input("Name an artist: "))
#rihanna = ArtistProfile("rihanna")
#
#rihanna.GetListUrls()
#
#for list_page in rihanna.list_urls:
#    rihanna.GetSongUrls(list_page)
#
#rihanna.GetLyrics()


