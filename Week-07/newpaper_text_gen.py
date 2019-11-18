"""

Prodictive text
text completion
"""


from newspaper import Article
import numpy as np
import pandas as pd
def article_text(url):
    art=Article(url)
    art.download()
    art.parse()
    text = art.text.split()
    return text

text1 = article_text('https://www.wikizeroo.org/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvU3Rhcl9XYXJz')

#make pairs
def make_pairs(text):
    pairs = []
    for i in range(len(text)-1):
        pairs.append([text[i], text[i+1]])
    return pairs

pairs = make_pairs(text1)

print(pairs[:5])
#to dict
word_dict = {}
for word1, word2 in pairs:
    if word1 in word_dict.keys():
        word_dict[word1].append(word2)
    else:
        word_dict.add([word1]:[word2]) ## not sure

for i in range(n)
