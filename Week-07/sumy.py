from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
#!python -c "import nltk; nltk.download('punkt')"
file = 'yo1.txt'
LANGUAGE = 'english'
COUNT = 10

parser = PlaintextParser.from_file(file, Tokenizer|LANGUAGE)



from sumy.summarizers.text_rank import TextRankSummarizer
"""
Tokenize the sentences then find cosin similartity.
What's similar to another.
Then rank sentences on highest similarity
and return the COUNT number you want.
"""
tsa = TextRankSummarizer()
tsa1 = tsa(parser.document, COUNT)

for i in tsa1:
    print(i)
    print("/n")


from sumy.summarizers.lsa import LsaSummarizer
"""
Latent Semantic Analysis
Supervised and Unsupervised learning.
Great for non-fiction summaries
"""
lsa = TextRankSummarizer()
lsa1 = lsa(parser.document, COUNT)

for i in lsa1:
    print(i)
    print("/n")


from sumy.summarizers.edmundson import EdmundsonSummarizer
"""
Supervised and Unsupervised learning.
Apply weights to words/sentences
"""
esa = TextRankSummarizer()
esa.bonus_words = ['supervised','unsupervised', 'machine'] #add extra weight to these words
esa.stigma_words = ['another', 'and'] #reduce the weight of these words
esa.null_words = ['another', 'and'] #reduce weights
esa1 = esa(parser.document, COUNT)

for i in esa1:
    print(i)
    print("/n")
