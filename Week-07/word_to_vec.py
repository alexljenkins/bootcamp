import spacy

model = spacy.load('en_core_web_md')

king = model.vocab['king'].vector

queen = model.vocab['queen'].vector
man = model.vocab['man'].vector
woman = model.vocab['woman'].vector
pretend_queen = king - man + woman

from sklearn.metrics.pairwise import cosine_similarity

vectors = []

vectors.append(king)
vectors.append(queen)
vectors.append(man)
vectors.append(woman)
vectors.append(pretend_queen)

import pandas as pd
df = pd.DataFrame(cosine_similarity(vectors).round(2),index = ['king','queen','man','woman','pretend_queen'],columns = ['king','queen','man','woman','pretend_queen'])
print(df)

# gensim python package builds a Workd2Vec models you can train yourself
# spacy has prebuilt
# vader does everything / sentiment