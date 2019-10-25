# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:48:16 2019

@author: alexl
"""

import spacy
model = spacy.load('en_core_web_md')



corpus = ["This is the real life",
          "Not a fantacy",
          "there's so much we can do",
          "but who has the time to fly"]

token_corpus = []

#created a list of words that are actually token corpi that have additional properties
for string in corpus:
    doc_string = model(string)
    token_corpus.append(doc_string)
    
print(token_corpus)

for word in token_corpus[0]:
    print(word, word.lemma_,
          word.is_stop,  # Bool is it a stop word?
          word.pos_,  # part of speech
          word.is_alpha  # alphabetical or numerical
          )

spacy.explain('ADP')  # explores meaning

def clean(song):
    doc = model(song)
    clean_text = ''
    for word in doc:
        if not word.is_stop and word.lemma_ != '-PRON-' and word.pos_ != 'PUNCT':
            word = word.lemma_
            clean_text += word + ' '
    return clean_text

