"""
Really quickly building out own sentiment analysis on
feedback strings of the previous lecture.
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import backend as K
K.clear_session()

reviews = ["I thought the lecture was rubbish and it overran a little on time",
          "Interesting content, nicely designed slides and learned something new about graph databases",
          'Very nice presentation with cute looking slides' ,
          "This mornings lecture on Graph was quite interesting but not detailed enough on the how",
          "interesting lecture, liked the neo4j panama paper example",
          "I liked the neo4j part but I hated how long the garage video was",
          "I enjoy learning about small innovation entities in companies and here how they function - cool to see a data science specific version",
          'Christo gave a very good presentation but he is just no Tom',
          "I don't ever want to work for Cap Gemini, after hearing all that rubbish today. What a waste of time. Happy weekend!",
          "The lecture was terribly informative. :smile: There was so much to learn and the graph database was crazy cool",
          "The presentation was good, graph theory is quite interesting"
          ]

labels = [0,1,1,1,1,0,1,0,0,1,1]

#%%
# 1. Create a vocab_to_keys and keys_to_vocab list for each unique word in the data

vocab = []
max_length = 0

#create a list of unique words from the full dataset
for review in reviews:
    review = review.lower().split()
    for word in review:
        vocab.append(word)
    if len(review) > max_length:
        max_length = len(review)

vocab = list(set(vocab))
max_length += 1

# create a number value for each word in the list
vocab_to_keys = {}
keys_to_vocab = {}

for i in range(len(vocab)):
    vocab_to_keys[vocab[i]] = i+1
    keys_to_vocab[i+1] = vocab[i]

#%%
# 2. Integer encode the words in each document
numerical_review = [[vocab_to_keys[x] for x in review.lower().split()] for review in reviews]

# 3. make each review equal in number length (words)
padded_reviews = sequence.pad_sequences(numerical_review, maxlen = max_length, padding = "pre")

# 4. Build the model
model = Sequential()

model.add(Embedding(input_dim=len(vocab)+1, output_dim=64)) # normalizes the data
model.add(LSTM(units = 64)) #units is describing the number of dimensions of the cell state | might need to add a dropout/reoccurent_dropout
model.add(Dense(1, activation = "sigmoid"))

model.summary()

# Compile and fit on training data
model.compile(optimizer='adam', loss = "binary_crossentropy", metrics=["accuracy"])

X = padded_reviews
y = np.array(labels)

model.fit(X, y, epochs=50)


# from tensorflow.keras.datasets import imdb # good model to train sentiment models on

# 5. New data/review
new_review = ["the lecture was a total joke nothing good will come of this",
                "you don't know what you're talking about it was amazing"]
# only select the words in the vocab list already, and encode them into numbers
nr = [[vocab_to_keys[x] for x in review.lower().split() if x in vocab] for review in new_review]
# pad again to make it the same length
pr = sequence.pad_sequences(numerical_review, maxlen = max_length, padding = "pre")

ypred = model.predict(pr)

print(ypred)
