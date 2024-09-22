from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from wordcloud import WordCloud, STOPWORDS

# Ensure the required packages are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define custom stopwords
custom_stopwords = set(['music', 'listen', 'hear', 'remember', 'song'])
all_stopwords = set(STOPWORDS)
all_stopwords.update(custom_stopwords)
# print(all_stopwords)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    print(nltk.pos_tag([word])[0])
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                "V": wordnet.VERB, "R": wordnet.ADV, "S": wordnet.ADJ_SAT}
    print(tag_dict.get(tag, wordnet.NOUN))
    return tag_dict.get(tag, wordnet.NOUN)


def pre_process(memory_text: str):
    # Tokenize the text
    words = word_tokenize(memory_text)

    # Remove punctuation and stopwords, and lemmatize the words
    lemmatized_words = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word))
                        for word in words]

    print(' '.join(lemmatized_words))

    lemmatized_words = [
        word for word in lemmatized_words if word not in all_stopwords and word not in string.punctuation]

    # Join the lemmatized words back into a string
    cleaned_text = ' '.join(lemmatized_words)
    print(cleaned_text)
    return cleaned_text


# Example text
text_data = 'listening to the always heard angrily beautifully excellently songs when I was in 8th interestingly graded and more better watched the melancholy music videos, a heavy dogs died in it and it was smooth sad.'
text_data = 'me and my friends singing this song when it came up on the radio in my friends car we were going to the mall in the summer last summer it was a hot day and i think we were going to the mall to shop for a gift for one of my other friends'
text_data = 'when this song got popular I liked it sorta but I had a grudge against it because bastille was my favourite artist and I felt like they had much better songs that deserved the spotlight more. I expressed this sentiment both my dad and some school friends who all agreed'
text_data = 'This song got me through my break up. I used to scream sing it while i did school work, took a shower, literally anything'
text_data = 'This song reminded me of when my friends and I went to my cottage one weekend in the summer. This song was very popular at the time and we kept hearing it everywhere we went. Eventually we all got sick of this song and got frustrated whenever we heard it.'
text_data = 'this song is on my cottage playlist which I would put on while boating or hanging around at the cottage. Hearing the song makes me thing of summer and the hot weather and smell of being on the lake'

# Pre-process the text
pre_process(text_data)
