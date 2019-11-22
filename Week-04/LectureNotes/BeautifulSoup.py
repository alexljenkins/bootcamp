# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:00:38 2019

@author: alexl
"""

import requests
from bs4 import BeautifulSoup as soup


rihanna = requests.get('https://www.metrolyrics.com/rihanna-lyrics.html')
#looking for a 200 responce being ok

#View the basic text of the request
rihanna.text

#convert it to html (source) format
rihanna_soup = soup(rihanna.text, 'html.parser')
rihanna_soup

#returns everything inside the div class
song_tag = rihanna_soup.find_all('div')

#returns everything with the class:name
#song_tag2 = rihanna_soup.find_all(attrs = {"class":"title hasvidtable"})
song_tag2 = rihanna_soup.find_all(attrs = {"class":"module"})
song_tag2

print(type(song_tag2))

#going into soup deeper and deeper and returning the urls inside the class:module
for each in song_tag2:
    print(each.find_all('a')[0].get('href'))


