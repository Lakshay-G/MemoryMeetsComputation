# MemoryMeetsComputation

This repository consists of all the code required by the paper []. This paper provides a compehensive study for sentiment analysis for psychology data. Currently we are comparing existing methods of [VADER](https://github.com/cjhutto/vaderSentiment) and [TextBlob](https://github.com/sloria/TextBlob) for two different datasets. Briefly, we're doing 4 things: (a) sentiment analysis (b) metrics for the sentiment analysis (c) wordcloud generation (d) top word counts; in this code.

## Getting Started

## Installation

1. Clone this repository using
   `git clone https://github.com/Lakshay-G/MemoryMeetsComputation.git`
2. Install the required _Python_ libraries using
   `pip install -r requirements.txt`
3. Download **snowball stopwords list** from [data_stopwords_snowball.rda](https://github.com/quanteda/stopwords/blob/master/data/data_stopwords_snowball.rda)

## Repository Sitemap

1. [asset/](asset/): Datasets needed to be analyzed and stopwords list
2. [output/](output/): All the generated output files
3. [src/](src/): Main code files for analysis and output generation
   - [archive/](src/archive/): Contains old code used during testing
   - [preprocess.py](src/preprocess.py): Contains complete preprocessing pipeline used before analysis
   - [sentiment_analysis.py](src/sentiment_analysis.py): Contains code for sentiment analysis as a form of histograms and confusion matrices
   - [snowball_stopwords_extract.R](src/snowball_stopwords_extract.R): Contains _R_ code to extract the stopwords list
   - [word_cloud.py](src/word_cloud.py): Contains code for word clouds and top 10 word counts plots
4. [LICENSE](LICENSE): Licence for the project
5. [README.md](README.md): Documentation for the project
6. [requirements.txt](requirements.txt): Python dependencies

## Usage

2. Since [data_stopwords_snowball.rda](https://github.com/quanteda/stopwords/blob/master/data/data_stopwords_snowball.rda) is an _R_ object, we need to extract the stopwords list for using it in _Python_.

   - Download [data_stopwords_snowball.rda](https://github.com/quanteda/stopwords/blob/master/data/data_stopwords_snowball.rda) document and keep in `assets` folder.

   - In _R Studio_, run the `src/snowball_stopwords_extract.R` file to get `stopwords_en.txt`. This is the `.txt` version of the snowball stopwords list in English language.

   - This list still contains some negators which are not ideal for sentiment analysis. Hence, we choose to remove: `["isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't","hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't", "against", "no", "nor", "not"]` words from the text file manually. One should go ahead and delete them as well.

   - Also, this stopwords list doesn't contain the token `'s` which is usually present in other stopwords list. Hence, we add `'s` token to the stopwords list as well.

3. The parameters needed for the code globally are saved in `params.json`.

   - `"data"`: contains parameters needed for extracting values

     - `"all_memories_path"`: contains the path to the excel file containing the dataset

     - `"stopwords_path"`: contains the path to the text file containing modified snowball stopwords list

     - `"sentiment_threshold"`: contains the threshold set for 3 bin sentiment classification between VADER and TextBlob for neutral statements

     - `"preprocess"`: contains `true` if preprocessing needs to be done before analysis, else `false`

     - `"sentiment_type"`: contains `"hist"` if sentiment analysis graphs need to be produced, else `"cm"` if sentiment analysis metrics and confusion matrices need to be produced

   - `"output"`: contains paths to all the output files; `wordcloud`, `wordcounts`, `sentiment histogram`, `confusion matrices` in the `output` folder.

4. Code for the entire preprocessing pipeline is `src/preprocess.py` script. <<NEED TO ADD PREPROCESSING STEPS>>
