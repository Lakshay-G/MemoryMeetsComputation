# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

if __name__ == '__main__':
    print('Hello')

    # # Sample memory text
    # memory_text = "I had a wonderful time at the park."

    # # Create a TextBlob object
    # blob = TextBlob(memory_text)

    # # Get sentiment
    # sentiment = blob.sentiment
    # print(
    #     f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")
    # Sample memory text
    memory_text = "I had a time at the park which helped me enjoy a lot."

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Get sentiment scores
    sentiment_scores = analyzer.polarity_scores(memory_text)
    print(sentiment_scores)
