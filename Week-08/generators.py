"""

GENERATORS:
inside the random number generator
zip()
enumerate()
open()
itertools library
df.iterrows()
results from SQLAlchemy/SQLite query
Scrapy Spiders
Out-of-Membory DataFrame() (dask library)
Spark (MapReduce paradigm - a king-size generator for distributed computing)

"""

def numbers():
    return[1,2,3]


def numbers2():
    for x in range (1,11):
        yield x #yield makes a function a generator


def numbers3():
    i = 1
    while True:
        yield i**2
        i +=1  # doesn't matter that the while is infinit. the generator is lazy

g = numbers2()
h = numbers3()
#%%



#%% list comprehension
squares_list = [i*i for i in range(1,11)] #held in memory
# create the same but as a generator
squares_gen = (i*i for i in range(1,11))
# or to make it go forever:
def integers():
    i = 1
    while True:
        yield i += 1

squares_gen = (i*i for i in integers())

print(g) # generator object

next(g) # pulls the next number in the generator


"""
RELATED CONCEPTS

iterator: generator generated from eg a list.
iterable: data structure you can use in for or an interator

"""


#very simple markov chain:

def markov():
    states = ['breakfast','lunch','dinner']
    i = 0
    while True:
        yield states[i]
        if i == 2:
            i = 0
        else:
            i+=1

m = markov()
[next(m) for i in range(7)]
