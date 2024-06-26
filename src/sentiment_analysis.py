# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import random

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

def sentiment_histogram(vader_scores, textblob_scores, cnt, sentiment_output_path, cue_val=None, cue_type=None):
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
    
    ax.set_xlabel('Polarity values')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Saving the plot
    if cue_type != None:
        ax.set_title(f'Cue value: {cue_val}, # memories: {cnt}')
        plt.savefig(
            f'{sentiment_output_path}/{cue_type}/{cue_val}_{cnt}.png', format='PNG')
    else:
        ax.set_title(f'All Memories, # memories: {cnt}')
        plt.savefig(
            f'{sentiment_output_path}/All_Memories_{cnt}.png', format='PNG')

    # plt.show()

def polarity_score(score, sentiment_threshold):
    if score > sentiment_threshold:
        return 1
    elif score < (-1 * sentiment_threshold):
        return -1
    else:
        return 0

def sentimentByCueType(cue_type: str, df: pd.DataFrame, sentiment_output_path: str):
    
    # Find the unique cue values for a given cue type
    unique_cue_values = df[cue_type].unique()
    print(f'Unique cues :: \n{unique_cue_values}')

    for cue_val in unique_cue_values:

        # Find the data of a particular cue value for a particular cue type
        # Extract the 'Memory_Text' column
        # Drop NaN values and convert to a list
        filtered_df = df[df[cue_type] == cue_val]
        memory_texts = filtered_df['Memory_text'].dropna().tolist()
        cnt = len(memory_texts)

        vader_scores = []
        textblob_scores = []
        for memory in memory_texts:
            _, _, _, vader = vader_score(memory)
            txtblob, _ = textblob_score(memory)
            vader_scores.append(polarity_score(vader, sentiment_threshold))
            textblob_scores.append(polarity_score(txtblob, sentiment_threshold))

        sentiment_histogram(vader_scores=vader_scores, textblob_scores=textblob_scores, cnt=cnt, sentiment_output_path=sentiment_output_path, cue_val=cue_val, cue_type=cue_type)

def sentimentOverall(df: pd.DataFrame, sentiment_output_path: str):
    # Extract the 'Memory_Text' column
    # Drop NaN values and convert to a list
    # memory_texts = df['Memory_text'].dropna().tolist()
    # cnt = len(memory_texts)
    memory_texts = df['Memory_text'].dropna().tolist()
    cnt = len(memory_texts)

    vader_scores = []
    textblob_scores = []
    for memory in memory_texts:
        _, _, _, vader = vader_score(memory)
        txtblob, _ = textblob_score(memory)
        vader_scores.append(polarity_score(vader, sentiment_threshold))
        textblob_scores.append(polarity_score(txtblob, sentiment_threshold))

    sentiment_histogram(vader_scores=vader_scores, textblob_scores=textblob_scores, cnt=cnt, sentiment_output_path=sentiment_output_path)

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

    random.seed(seed_value)
    np.random.seed(seed_value)

    # Find and print the unique values from the cue type: [Song, Condition, Year, Singer]
    # cue_type = 'Singer'
    # for cue_type in ['Song', 'Singer', 'Year', 'Condition']:
    for cue_type in ['Year']:
        print()
        sentimentByCueType(cue_type=cue_type, df=df, sentiment_output_path=sentiment_output_path)

    sentimentOverall(df=df, sentiment_output_path=sentiment_output_path)