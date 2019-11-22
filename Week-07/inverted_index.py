from sklearn.datasets import fetch_20newsgroups

corpus = fetch_20newsgroups()['data']
print(len(corpus))


#%%
query = 'data'

def brute_force_query(query: str, corpus: list): #type annotations
    """ Returns indices of matching documents to query.
    Takes 2.5 seconds to run. O(n) """
    for i, doc in enumerate(corpus):
        for word in doc.split():
            if word == query:
                yield i #generator. allows returns 1 by 1



# %time r = brute_force_query('data',corpus)
# %timeit r = brute_force_query('data',corpus)

"""
Inverted index
"""
def construct_index(corpus): #3 sections to build the index (only need to do this once)
    """
    takes O(n) to build. but O(1) to search afterwards.
    """
    index = {}
    for i, doc in enumerate(corpus):
        for word in doc.split():
            if word not in index:
                index[word] = [i]
            else:
                index[word].append(i)
    return index

#%timeit index = construct_index(corpus)
index = construct_index(corpus)
index['data'] #94 nano seconds to run. O(1) time or O(log(n)).


def construct_index_v2(corpus): #3 sections to build the index (only need to do this once)
    """
    takes O(n) to build. but O(1) to search afterwards.
    """
    # from collections import defaultdict
    # index = defaultdict(list) #special kind of dic
    index = {}
    for i, doc in enumerate(corpus):
        for word in doc.split():
            index.setdefault(word, []) #creates a default dic val type
                index[word].append(i)
    return index


query = "great data science"
results = index['great'] + index['data'] + index['science']

""" remember this collection"""
from collections import Counter
c = Counter(results)
c.most_common(3)


### Other things to consider:
# ranking might require normalisation for corpus length.
# typos: Levenshtein distance (partical miss-matches)
# bigger typos: trigram search (heuristic algorithm)
# Standard NLP: language, tokenize, stop words, remove crap like numbers?
# text importance for titles/links vs body

### Technologies:
# Postgres
# mongo_db
# Elasticsearch (for big text searches)
