from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
import nltk
import numpy as np
import pandas as pd
from wordcloud import STOPWORDS


def pre_process(memory_text: str):

    print(memory_text)
    return


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Example text
text_data = """
Listened to the song when I was in 8th grade and watched the music video, a dog died in it and it was sad.
"""

# Define custom stopwords
custom_stopwords = set([' '])
print(custom_stopwords)

# Combine default NLTK stopwords with custom stopwords
# all_stopwords = set(stopwords.words('english')).union(custom_stopwords)
# print(all_stopwords)

all_stopwords = set(STOPWORDS)
all_stopwords.update(custom_stopwords)
print(all_stopwords)


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to convert NLTK POS tags to WordNet POS tags


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Step 1: Remove sentences with less than 10 words
sentences = sent_tokenize(text_data)
filtered_sentences = [sentence for sentence in sentences if len(
    word_tokenize(sentence)) >= 10]

# Step 2: Remove stopwords (both default and custom) and punctuation


def remove_stopwords_and_punctuation(sentence):
    words = word_tokenize(sentence)
    words = [word for word in words if word.lower(
    ) not in all_stopwords and word not in string.punctuation]
    return words


filtered_sentences = [' '.join(remove_stopwords_and_punctuation(
    sentence)) for sentence in filtered_sentences]

# Step 3: Lemmatize the words with POS tagging


def lemmatize_sentence(sentence):
    words = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(
        word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized_words)


lemmatized_sentences = [lemmatize_sentence(
    sentence) for sentence in filtered_sentences]

# Step 4: Convert to lower case
final_sentences = [sentence.lower() for sentence in lemmatized_sentences]

# Display the preprocessed text
preprocessed_text = ' '.join(final_sentences)
print(preprocessed_text)
