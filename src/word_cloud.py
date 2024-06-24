# Importing necessary libraries
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import json


def wordCloudByCueType(cue_type: str, df: pd.DataFrame, stopwords: list):
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
        # Extract the 'Memory_Text' column
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
        plt.title(f'Cue type: {cue_val}, # memories: {cnt}')

        # Remove ticks and labels on both axes
        plt.xticks([])  # Remove ticks on the x-axis
        plt.yticks([])  # Remove ticks on the y-axis

        plt.savefig(
            f'{wordcloud_output_path}/{cue_type}/{cue_val}_{cnt}.png', format='PNG')
        # plt.show()
        plt.close()
    return wordcloud


def wordCloudOverall(df: pd.DataFrame, stopwords: list):
    '''
    df: The entire data frame for a given dataset
    stopwords: particular words we don't need to be included in the word cloud
    returns: The function returns the word cloud if needed in future. 
             Also, this function saves the word cloud image in the WordClouds folder.
    '''

    # Extract the 'Memory_Text' column
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
    return wordcloud


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
    sentiment_output_path = param['output']['sentiment_output_path']

    # Read the excel file for the data
    df = pd.read_excel(data_file_path)
    df_columns = df.columns.to_list()
    print(f'Data columns are :: \n{df_columns}')

    # Find and print the unique values from the cue type: [Song, Condition, Year, Singer]
    # cue_type = 'Singer'
    for cue_type in ['Song', 'Singer', 'Year', 'Condition']:
        wordCloudByCueType(cue_type=cue_type, df=df, stopwords=stopwords)

    wordCloudOverall(df=df, stopwords=stopwords)
