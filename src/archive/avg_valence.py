
# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import random


def vader_score(text: str):
    '''
    text: Takes in the sentence/ bunch of sentences to compute the negativity, positivity, neutrality and compunded scores
    returns: negative, neutral, positive and compunded scores

    After using VADER we will get  4 values: neg, neu, pos, compound.
    Compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
    '''

    vader_sentiment = SentimentIntensityAnalyzer()
    score = vader_sentiment.polarity_scores(text)
    return score['neg'], score['neu'], score['pos'], score['compound']


def textblob_score(text: str):
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


def sentiment_histogram(vader_scores: list, textblob_scores: list, cnt: int, sentiment_output_path: str, cue_val=None, cue_type=None):
    '''
    vader_scores: Takes in polarity scores generated from Vader method
    textblob_scores: Takes in polarity scores generated from Textblob method
    cnt: Count of each memories of a particular type
    sentiment_output_path: Output path for the plots
    cue_type: None (by default) if the dataset is not separated by cue types. If it is separated, enter the cue_type.
    cue_val: None (by default) if the dataset is not separated by cue types. If it is separated, enter the individual cue_val from the cue_type.
    The function saves a histogram plot in the respective output location.
    '''
    bins = np.arange(-1, 3)

    # Creating the histogram data
    hist_vader, _ = np.histogram(vader_scores, bins=bins)
    hist_textblob, _ = np.histogram(textblob_scores, bins=bins)

    # Setting the width of each bar
    width = 0.35

    # Creating the figure and axes
    fig, ax = plt.subplots()

    # Plotting the histograms
    ax.bar(bins[:-1] - width/2, hist_vader, width=width,
           label='VADER analysis', color='blue')
    ax.bar(bins[:-1] + width/2, hist_textblob, width=width,
           label='Textblob analysis', color='orange')

    # Adding titles and labels
    ax.set_xlabel('Polarity values')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Saving the plot
    if cue_type != None:
        ax.set_title(
            f'Cue type: {cue_type}, cue value: {cue_val}, # memories: {cnt}')
        plt.savefig(
            f'{sentiment_output_path}/{cue_type}/{cue_val}_{cnt}.png', format='PNG')
    else:
        ax.set_title(f'All Memories, # memories: {cnt}')
        plt.savefig(
            f'{sentiment_output_path}/All_Memories_{cnt}.png', format='PNG')

    # plt.show()
    plt.close()


def polarity_score(score: float):
    if score < -0.6:
        return 1
    elif score < -0.2:
        return 2
    elif score < 0.2:
        return 3
    elif score < 0.6:
        return 4
    else:
        return 5


def avg_Score(cue_type):
    unique_cue_values = df[cue_type].unique()
    print(f'Unique cues :: \n{unique_cue_values}')

    selfvalence_data = []
    vader_data = []
    textblob_data = []

    for cue_val in unique_cue_values:
        vader_scores = []
        textblob_scores = []

        filtered_df = df[df[cue_type] == cue_val]
        memory_texts = filtered_df['text']
        for memory in memory_texts:
            _, _, _, vader = vader_score(memory)
            txtblob, _ = textblob_score(memory)
            vader_scores.append(polarity_score(vader))
            textblob_scores.append(polarity_score(txtblob))

        avg = np.average(np.array(filtered_df['Valence']))
        vader_avg = np.average(np.array(vader_scores))
        textblob_avg = np.average(np.array(textblob_scores))
        print(avg, vader_avg, textblob_avg)
        print(len(filtered_df['Valence']), len(
            vader_scores), len(textblob_scores))

        selfvalence_data.append(np.array(filtered_df['Valence']))
        vader_data.append(np.array(vader_scores))
        textblob_data.append(np.array(textblob_scores))

    print('Self valence score: ', selfvalence_data)
    print('VADER score: ', vader_data)
    print('Textblob score: ', textblob_data)
    plt.boxplot(selfvalence_data, labels=['Lyrics', 'Music', 'Song ', 'Name'])
    plt.title('Self_Valence')
    # plt.ylim(1, 5)
    plt.show()

    plt.boxplot(vader_data, labels=['Lyrics', 'Music', 'Song ', 'Name'])
    plt.title('VADER')
    plt.show()

    plt.boxplot(textblob_data, labels=['Lyrics', 'Music', 'Song ', 'Name'])
    plt.title('TEXTBLOB')
    plt.show()


def sentimentByCueType(cue_type: str, df: pd.DataFrame, sentiment_output_path: str):
    '''
    cue_type: [Song, Condition, Year, Singer] are the examples in the given work
    df: The entire data frame for a given dataset
    sentiment_output_path: Output path for the plots
    The function returns the saves the histogram for the different cue values in a given cue type.
    '''

    # Find the unique cue values for a given cue type
    unique_cue_values = df[cue_type].unique()
    print(f'Unique cues :: \n{unique_cue_values}')

    for cue_val in unique_cue_values:

        # Find the data of a particular cue value for a particular cue type
        # Extract the 'Text' column
        # Drop NaN values and convert to a list
        filtered_df = df[df[cue_type] == cue_val]
        memory_texts = filtered_df['text'].dropna().tolist()
        cnt = len(memory_texts)

        vader_scores = []
        textblob_scores = []
        for memory in memory_texts:
            _, _, _, vader = vader_score(memory)
            txtblob, _ = textblob_score(memory)
            vader_scores.append(polarity_score(vader, sentiment_threshold))
            textblob_scores.append(polarity_score(
                txtblob, sentiment_threshold))

        sentiment_histogram(vader_scores=vader_scores, textblob_scores=textblob_scores, cnt=cnt,
                            sentiment_output_path=sentiment_output_path, cue_val=cue_val, cue_type=cue_type)


def sentimentOverall(df: pd.DataFrame, sentiment_output_path: str):
    '''
    df: The entire data frame for a given dataset
    sentiment_output_path: Output path for the plots
    The function returns the saves the histogram for the sentiment analysis in all of dataset.
    '''

    # Extract the 'Text' column
    # Drop NaN values and convert to a list
    # memory_texts = df['text'].dropna().tolist()
    # cnt = len(memory_texts)
    memory_texts = df['text'].dropna().tolist()
    cnt = len(memory_texts)

    vader_scores = []
    textblob_scores = []
    for memory in memory_texts:
        _, _, _, vader = vader_score(memory)
        txtblob, _ = textblob_score(memory)
        vader_scores.append(polarity_score(vader, sentiment_threshold))
        textblob_scores.append(polarity_score(txtblob, sentiment_threshold))

    sentiment_histogram(vader_scores=vader_scores, textblob_scores=textblob_scores,
                        cnt=cnt, sentiment_output_path=sentiment_output_path)


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

    # Ensures to get the repeated results
    random.seed(seed_value)
    np.random.seed(seed_value)

    # Find and print the unique values from the cue type: [Song, Condition, Year, Singer]
    # cue_type = 'Singer'
    for cue_type in ['Condition']:
        # for cue_type in ['Song', 'Singer', 'Year', 'Condition']:
        print(cue_type)
        # sentimentByCueType(cue_type=cue_type, df=df,
        #                    sentiment_output_path=sentiment_output_path)
        avg_Score(cue_type)

    # sentimentOverall(df=df, sentiment_output_path=sentiment_output_path)
