# MemoryMeetsComputation

This repository consists of all the code required by the paper []. This paper provides a compehensive study for sentiment analysis for psychology data. Currently we are comparing existing methods of VADER and TextBlob for two different datasets. Briefly, we're doing 4 things: (a) sentiment analysis (b) metrics for the sentiment analysis (c) wordcloud generation (d) top word counts; in this code.

## Getting Started

## Installation

1. Clone this repository using
   `git clone https://github.com/Lakshay-G/MemoryMeetsComputation.git`
2. Install the required Python libraries using
   `pip install -r requirements.txt`
3. Download **snowball stopwords list** from [data_stopwords_snowball.rda](https://github.com/quanteda/stopwords/blob/master/data/data_stopwords_snowball.rda)

## Usage

1. Since [data_stopwords_snowball.rda](https://github.com/quanteda/stopwords/blob/master/data/data_stopwords_snowball.rda) is an _R_ object, we need to extract the stopwords list for using it in _Python_.

   - Download [data_stopwords_snowball.rda](https://github.com/quanteda/stopwords/blob/master/data/data_stopwords_snowball.rda) document and keep in `assets` folder.
   - In _R Studio_, run the `src/snowball_stopwords_extract.R` file to get `stopwords_en.txt`. This is the `.txt` version of the snowball stopwords list in English language.
   - This list still contains some negators which are not ideal for sentiment analysis. Hence, we choose to remove: \n
     `["isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't","hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't", "against", "no", "nor", "not"]` words from the text file manually. One should go ahead and delete them as well.
   - Also, this stopwords list doesn't contain the token `'s` which is usually present in other stopwords list. Hence, we add `'s` token to the stopwords list as well.

2. The parameters needed for the code globally are saved in `params.json`.
