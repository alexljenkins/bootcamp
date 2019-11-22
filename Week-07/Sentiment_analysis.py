from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

analyzer.polarity_scores("your mum is so fat that when she jumped for joy, she got stuck :'(")
analyzer.polarity_scores("</3")
neg, neu, pos, com = analyzer.polarity_scores("</3").values()
neg
