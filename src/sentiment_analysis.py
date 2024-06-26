# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

def vader_score(text):
    '''
    text: Takes in the sentence/ bunch of sentences to compute the negativity, positivity, neutrality and compunded scores
    returns: negative, neutral, positive and compunded scores

    After using VADER we will get  4 values: neg, neu, pos, compound.
    Compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
    '''
    
    vader_sentiment = SentimentIntensityAnalyzer()
    score = vader_sentiment.polarity_scores(text) 
    return score['neg'], score['neu'], score['pos'], score['compound']

def textblob_score(text):
    '''
    text: Takes in the sentence/ bunch of sentences to compute the polarity and subjective scores
    returns: polarity and subjective scores
    
    textblob_sentiment.sentiment will give us 2 values: polarity and subjectivity
    The polarity score is a float within the range [-1.0, 1.0] where -1.0 is very negative and 1.0 is very positive.
    The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    '''
    
    textblob_sentiment = TextBlob(text)
    score_polarity = textblob_sentiment.sentiment.polarity
    score_subjectivity = textblob_sentiment.sentiment.subjectivity
    return score_polarity, score_subjectivity

def sentiment_histogram(vader_scores, textblob_scores, cnt):
    bins = np.arange(-1, 3)

    # Creating the histogram data
    hist_vader, _ = np.histogram(vader_scores, bins=bins)
    hist_textblob, _ = np.histogram(textblob_scores, bins=bins)

    # Setting the width of each bar
    width = 0.35

    # Creating the figure and axes
    fig, ax = plt.subplots()

    # Plotting the histograms
    ax.bar(bins[:-1] - width/2, hist_vader, width=width, label='VADER analysis', color='blue')
    ax.bar(bins[:-1] + width/2, hist_textblob, width=width, label='Textblob analysis', color='orange')

    # Adding titles and labels
    ax.set_title(f'Cue value: 7 Rings, # memories: {cnt}')
    ax.set_xlabel('Polarity values')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Showing the plot
    plt.show()

def polarity_score(score, sentiment_threshold):
    if score > sentiment_threshold:
        return 1
    elif score < (-1 * sentiment_threshold):
        return -1
    else:
        return 0

if __name__ == '__main__':

    # Open the parameter file to get necessary parameters
    param_filename = 'src/params.json'
    with open(param_filename) as paramfile:
        param = json.load(paramfile)

    # Load parameter values
    seed_value = param['data']['seed_value']
    data_file_path = param['data']['all_memories_path']
    stopwords = param['data']['stopwords']
    sentiment_threshold = param['data']['sentiment_threshold']
    sentiment_output_path = param['output']['sentiment_output_path']

    # Read the excel file for the data
    df = pd.read_excel(data_file_path)
    df_columns = df.columns.to_list()
    print(f'Data columns are :: \n{df_columns}')

    # Extract the 'Memory_Text' column
    # Drop NaN values and convert to a list
    # memory_texts = df['Memory_text'].dropna().tolist()
    # cnt = len(memory_texts)
    filtered_df = df[df['Song'] == '7 Rings']
    memory_texts = filtered_df['Memory_text'].dropna().tolist()
    cnt = len(memory_texts)

    vader_scores = []
    textblob_scores = []
    for memory in memory_texts:
        _, _, _, vader = vader_score(memory)
        txtblob, _ = textblob_score(memory)
        vader_scores.append(polarity_score(vader, sentiment_threshold))
        textblob_scores.append(polarity_score(txtblob, sentiment_threshold))

        # print(f'sentence: {memory} \n VADER sentiment score: {vader} \n TextBlob score: {txtblob}')
        # print("=" * 30)

    sentiment_histogram(vader_scores=vader_scores, textblob_scores=textblob_scores, cnt=cnt)
