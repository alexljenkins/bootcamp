# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:00:38 2019
@author: alexl
"""

import requests
from bs4 import BeautifulSoup as soup
import re


regex = 'href=\"([^\"]+)\"'

#dotall makes newlines included


class ArtistProfile():
    
    def __init__(self, name):
        """
        Creates variables for the object
        """
        self.name = name.replace(" ", "-")
        self.song_urls = []
        self.song_lyrics = []
    
    def GetSongUrls(self):
        """
        Gets the URLs of the artist from metrolyrics
        """
        self.request = requests.get(f'https://www.metrolyrics.com/{self.name}-lyrics.html')
        self.request = soup(self.request.text, 'html.parser')
        self.request = self.request.find_all(attrs = {"class":"title hasvidtable"})
        
        for url in self.request:
            self.song_urls.append(url.get('href'))


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
            self.song_lyrics.append(song_request)
        
        
    def __repr__(self):
        
        return f"this is a return for debugging"


#ArtistProfile(input("Name an artist: "))
rihanna = ArtistProfile("rihanna")

rihanna.GetSongUrls()
print(f"We found {len(rihanna.song_urls)} number of songs by Rihanna")

rihanna.GetLyrics()
print(f"We saved {len(rihanna.song_lyrics)} number of lyrics from Rihanna")