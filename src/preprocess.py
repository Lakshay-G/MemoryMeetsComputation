import spacy
import pandas as pd
import numpy as np
import json
import time
from collections import Counter
import re


def lemmatizer(memory, model):
    # convert the memory given to tokenized sentence
    sentence = model(memory)

    # extract lemmatized form of all the tokens in the sentence
    # lemmata = [token.lemma_.lower() for token in sentence]
    lemmata = ['sing' if token.text.lower() == 'singing' else token.lemma_.lower()
               for token in sentence]  # tackling edge case since spacy model is lemmatizing 'singing' to 'singe' instead of 'sing'

    # join the lemmatized tokens to get the final lemmatized sentence
    final_sentence = ' '.join(lemmata)

    return final_sentence


def remove_stopwords(memory, stopwords):

    # Tokenize the memory (split into words)
    words = memory.lower().split()

    # Filter out stopwords
    filtered_words = [word for word in words if word not in stopwords]

    # Reconstruct the cleaned text
    cleaned_memory = ' '.join(filtered_words)

    return cleaned_memory


def find_custom_stopwords(memories_list):

    # combine all memories together to get custom stopwords
    memories = ' '.join(memories_list)

    # tokenize the combined memories
    tokens = memories.split()

    # count the frequency of each token and store it
    token_counts = Counter(tokens)
    frequencies = np.array(list(token_counts.values()))
    # print('frequency: ', sorted(frequencies, reverse=True),
    #       len(frequencies), np.sum(frequencies))
    mean_freq = np.mean(frequencies)  # mean of the frequency of tokens
    std_freq = np.std(frequencies)  # SD of the frequency of tokens

    # threshold of computing the custom stopwords is set to MEAN + 4*SD
    threshold = mean_freq + 4 * std_freq

    custom_stopwords = {word for word, count in token_counts.items()
                        if count > threshold}
    # Output results
    print(f"Token Frequencies (only showing Mean + 1 SD = {np.int32(mean_freq+std_freq)} tokens for now):",
          token_counts.most_common(np.int32(mean_freq+std_freq)))
    # print("Token Frequencies:", token_counts)
    print("Mean Frequency:", mean_freq)
    print("Standard Deviation:", std_freq)
    print('Threshold Frequency (Mean + 4 SDs): ', threshold)
    print("Custom Stopwords (4 SDs above mean):", custom_stopwords)

    return custom_stopwords


def remove_punctuation(memory):

    # Regex pattern to match all punctuation marks
    pattern = r"[^\w\s]"

    # Remove all punctuation marks
    memory_no_punctuation = re.sub(pattern, "", memory)

    return memory_no_punctuation


def preprocessing_pipeline(memories_list, stopwords):

    model = spacy.load(f'en_core_web_lg')

    print('\n\t\t >>>>>> STARTING PREPROCESSING STEPS <<<<<<\n\n')
    start = time.time()

    # 1. lemmatizing works good + lowering also done
    print("Step 1:: Starting lemmatization!\n")
    memories_list = [lemmatizer(memory, model) for memory in memories_list]

    # 2. now check for stopwords and remove the one's from snowball stopwords list
    print("Step 2:: Removing snowball stopwords (without negators and including 's token)!\n")
    memories_list = [remove_stopwords(memory, stopwords)
                     for memory in memories_list]

    # 3. remove all the punctuation marks now
    print("Step 3:: Removing punctuation marks!\n")
    memories_list = [remove_punctuation(memory) for memory in memories_list]

    # 4. now check for custom stopwords based on words occuring >3SDs away
    print("Step 4:: Finding custom stopwords!")
    custom_stopwords = find_custom_stopwords(memories_list)

    # 5. remove the custom stopwords now
    print("\nStep 5:: Removing custom stopwords now!\n")
    memories_list = [remove_stopwords(memory, custom_stopwords)
                     for memory in memories_list]

    print('FINAL:: Total time taken for all the preprocessing: ', time.time()-start)

    return memories_list


if __name__ == '__main__':
    # Open the parameter file to get necessary parameters
    param_filename = 'src/params.json'
    with open(param_filename) as paramfile:
        param = json.load(paramfile)

    # Load parameter values
    seed_value = param['data']['seed_value']
    data_file_path = param['data']['all_memories_path']
    # custom_stopwords = param['data']['stopwords']

    # Path to the TXT file
    file_path = 'asset/stopwords_en.txt'

    # Read the TXT file
    with open(file_path, 'r') as file:
        stopwords = file.read().splitlines()

    # Display the stopwords
    # print(stopwords)

    sentiment_threshold = param['data']['sentiment_threshold']
    sentiment_output_path = param['output']['sentiment_output_path']
    confusion_output_path = param['output']['confusion_output_path']

    # Read the excel file for the data
    df = pd.read_excel(data_file_path)
    memories_list = df['Memory_text'].to_list()

    memories_list = preprocessing_pipeline(memories_list, stopwords)
    # print(df['Memory_text'].head())
    df['Memory_text'] = memories_list
    # print(df.head())
