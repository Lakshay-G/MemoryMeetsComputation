# Importing necessary libraries
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np


def wordCountPlot(wordcloud: WordCloud, all_memory_texts: str, cnt: int, wordcount_output_path: str, cue_type=None, cue_val=None, sentiment=None):
    # Get the raw word counts from the word cloud
    raw_word_counts = wordcloud.process_text(all_memory_texts)

    # Get the top 10 most popular words by raw count
    top_10_words = sorted(raw_word_counts.items(),
                          key=lambda x: x[1], reverse=True)[:10]

    # This specific part counts cnns to make sure that we have total # of words for a given cue_type (excluding stopwords)
    '''raw_word_counts = sorted(raw_word_counts.items(),
                             key=lambda x: x[1], reverse=True)
    cnns = 0
    for _, cnn in raw_word_counts:
        cnns += cnn'''

    # Print the top 10 words with their raw counts
    words = []
    counts = []
    for word, count in top_10_words:
        # print(f"{word}: {count}")
        # sum += count
        words.append(word)
        counts.append(count)

    # print(sum)
    # print(words, counts)
    # print(sum(counts))
    plt.figure(figsize=(8, 5))
    plt.barh(words, np.array(counts) / sum(counts), color='orange')
    # plt.barh(words, np.array(counts) / cnns, color='orange')
    plt.ylabel('Words')
    plt.xlabel('Frequency of words (Probability)')
    plt.gca().invert_yaxis()
    if cue_type != None:
        plt.title(
            f'Top 10 words in Cue type: {cue_type}, cue value: {cue_val}, # memories: {cnt}')
        plt.tight_layout()
        if sentiment:
            plt.savefig(
                f'{wordcount_output_path}/{cue_type}/{cue_val}_{cnt}_{sentiment}.png', format='PNG')
        else:
            plt.savefig(
                f'{wordcount_output_path}/{cue_type}/{cue_val}_{cnt}.png', format='PNG')
    else:
        plt.title(f'Top 10 words in All Memories, # memories : {cnt}')
        plt.tight_layout()
        if sentiment:
            plt.savefig(
                f'{wordcount_output_path}/All_Memories_{cnt}_{sentiment}.png', format='PNG')
        else:
            plt.savefig(
                f'{wordcount_output_path}/All_Memories_{cnt}.png', format='PNG')

    plt.close()


def wordCloudByCueType(cue_type: str, df: pd.DataFrame, stopwords: list, seed_value: int, wordcloud_output_path: str, wordcount_output_path: str):
    '''
    cue_type: [Song, Condition, Year, Singer] are the examples in the given work
    df: The entire data frame for a given dataset
    stopwords: particular words we don't need to be included in the word cloud
    returns: The function returns the word cloud if needed in future. 
             Also, this function saves the word cloud image in the WordClouds folder.
    '''

    # Find the unique cue values for a given cue type
    unique_cue_values = df[cue_type].unique()
    print(f'Unique cues :: \n{unique_cue_values}')

    for cue_val in unique_cue_values:

        # Find the data of a particular cue value for a particular cue type
        # Extract the 'Memory_text' column
        # Drop NaN values and convert to a list
        filtered_df = df[df[cue_type] == cue_val]
        memory_texts = filtered_df['Memory_text'].dropna().tolist()
        cnt = len(memory_texts)

        # Combine all texts into a single string if needed
        all_memory_texts = ' '.join(memory_texts)

        # Create a set of stop words and add custom stop words
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(stopwords)

        wordcloud = WordCloud(width=800, height=400,
                              background_color='white', stopwords=custom_stopwords, colormap='viridis', collocations=False, random_state=seed_value).generate(all_memory_texts)

        # Display the generated word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(
            f'Cue type: {cue_type}, cue value: {cue_val}, # memories: {cnt}')

        # Remove ticks and labels on both axes
        plt.xticks([])  # Remove ticks on the x-axis
        plt.yticks([])  # Remove ticks on the y-axis

        plt.savefig(
            f'{wordcloud_output_path}/{cue_type}/{cue_val}_{cnt}.png', format='PNG')
        # plt.show()
        plt.close()

        wordCountPlot(wordcloud=wordcloud, all_memory_texts=all_memory_texts,
                      cnt=cnt, wordcount_output_path=wordcount_output_path, cue_type=cue_type, cue_val=cue_val)

    return wordcloud


def wordCloudOverall(df: pd.DataFrame, stopwords: list, seed_value: int, wordcloud_output_path: str, wordcount_output_path: str):
    '''
    df: The entire data frame for a given dataset
    stopwords: particular words we don't need to be included in the word cloud
    returns: The function returns the word cloud if needed in future. 
             Also, this function saves the word cloud image in the WordClouds folder.
    '''

    # Extract the 'Memory_text' column
    # Drop NaN values and convert to a list
    memory_texts = df['Memory_text'].dropna().tolist()
    cnt = len(memory_texts)

    # Combine all texts into a single string if needed
    all_memory_texts = ' '.join(memory_texts)

    # Create a set of stop words and add custom stop words
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(stopwords)

    wordcloud = WordCloud(width=800, height=400,
                          background_color='white', stopwords=custom_stopwords, colormap='viridis', collocations=False, random_state=seed_value).generate(all_memory_texts)

    # Display the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'All Memories, # memories: {cnt}')

    # Remove ticks and labels on both axes
    plt.xticks([])  # Remove ticks on the x-axis
    plt.yticks([])  # Remove ticks on the y-axis

    plt.savefig(
        f'{wordcloud_output_path}/All_Memories_{cnt}.png', format='PNG')
    # plt.show()
    plt.close()

    wordCountPlot(wordcloud=wordcloud, all_memory_texts=all_memory_texts,
                  cnt=cnt, wordcount_output_path=wordcount_output_path)

    return wordcloud


def wordCloudSentiments(df: pd.DataFrame, stopwords: list, seed_value: int, wordcloud_output_path: str, wordcount_output_path: str):
    '''
    df: The entire data frame for a given dataset
    stopwords: particular words we don't need to be included in the word cloud
    returns: The function returns the word cloud if needed in future. 
             Also, this function saves the word cloud image in the WordClouds folder.
    '''

    for sentiment in ['positive', 'negative', 'neutral']:

        if sentiment == 'positive':
            df_positive = df[df['Valence'] > 3]
        elif sentiment == 'negative':
            df_positive = df[df['Valence'] < 3]
        elif sentiment == 'neutral':
            df_positive = df[df['Valence'] == 3]

        # Extract the 'Memory_text' column
        # Drop NaN values and convert to a list
        memory_texts = df_positive['Memory_text'].dropna().tolist()
        cnt = len(memory_texts)

        # Combine all texts into a single string if needed
        all_memory_texts = ' '.join(memory_texts)

        # Create a set of stop words and add custom stop words
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(stopwords)

        wordcloud = WordCloud(width=800, height=400,
                              background_color='white', stopwords=custom_stopwords, colormap='viridis', collocations=False, random_state=seed_value).generate(all_memory_texts)

        # Display the generated word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'All Memories, # memories: {cnt}')

        # Remove ticks and labels on both axes
        plt.xticks([])  # Remove ticks on the x-axis
        plt.yticks([])  # Remove ticks on the y-axis

        plt.savefig(
            f'{wordcloud_output_path}/All_Memories_{cnt}_{sentiment}.png', format='PNG')
        # plt.show()
        plt.close()
        wordCountPlot(wordcloud=wordcloud, all_memory_texts=all_memory_texts,
                      cnt=cnt, wordcount_output_path=wordcount_output_path, sentiment=sentiment)
    # return wordcloud


if __name__ == '__main__':

    # Open the parameter file to get necessary parameters
    param_filename = 'src/params.json'
    with open(param_filename) as paramfile:
        param = json.load(paramfile)

    # Load parameter values
    seed_value = param['data']['seed_value']
    data_file_path = param['data']['all_memories_path']
    stopwords = param['data']['stopwords']
    wordcloud_output_path = param['output']['wordcloud_output_path']
    wordcount_output_path = param['output']['wordcount_output_path']

    # Read the excel file for the data
    df = pd.read_excel(data_file_path)
    df_columns = df.columns.to_list()
    print(f'Data columns are :: \n{df_columns}')

    # Find and print the unique values from the cue type: [Song, Condition, Year, Singer]
    # cue_type = 'Singer'
    for cue_type in ['Song', 'Singer', 'Year', 'Condition']:
        # for cue_type in ['Year']:
        wordCloudByCueType(cue_type=cue_type, df=df, stopwords=stopwords, seed_value=seed_value,
                           wordcloud_output_path=wordcloud_output_path, wordcount_output_path=wordcount_output_path)

    wordCloudOverall(df=df, stopwords=stopwords, seed_value=seed_value,
                     wordcloud_output_path=wordcloud_output_path, wordcount_output_path=wordcount_output_path)

    wordCloudSentiments(df=df, stopwords=stopwords, seed_value=seed_value,
                        wordcloud_output_path=wordcloud_output_path, wordcount_output_path=wordcount_output_path)
