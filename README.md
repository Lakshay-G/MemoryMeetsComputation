# MemoryMeetsComputation

This repository consists of all the code required by the paper []. This paper provides a compehensive study for sentiment analysis for psychology data. Currently we are comparing existing methods of [VADER](https://github.com/cjhutto/vaderSentiment) and [TextBlob](https://github.com/sloria/TextBlob) for two different datasets, [MEAM - Music Evoked Autobiographical Memories](asset/rIAMs_dataset_20250201.xlsx), and [rIAM - recurrent, Involuntary Autobiographical Memories](asset/rIAMs_dataset_20250201.xlsx). Briefly, we're doing 6 things: (a) sentiment analysis, (b) metrics for the sentiment analysis, (c) wordcloud generation, (d) top word counts, (e) ablation analysis, (f) optimal threshold analysis; in this code.

<details>
<summary><h2> Table of Contents </h2></summary>

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Repository Sitemap](#repository-sitemap)
4. [Setup](#setup)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Sentiment Analysis](#sentiment-analysis)
7. [Word Cloud and Word Counts](#word-cloud-and-word-counts)
8. [Ablation and Threshold Analysis](#ablation-and-threshold-analysis)
9. [Running the Analysis](#running-the-analysis)
</details>

## Installation

1. Clone this repository using
   ```bash
   git clone https://github.com/Lakshay-G/MemoryMeetsComputation.git
   ```
2. Install the required _Python_ libraries using
   ```bash
   pip install -r requirements.txt
   ```
3. Download **snowball stopwords list** from [data_stopwords_snowball.rda](https://github.com/quanteda/stopwords/blob/master/data/data_stopwords_snowball.rda)

## Repository Sitemap

1. [asset/](asset/): Datasets needed to be analyzed and stopwords list
2. [output/](output/): All the generated output files
3. [src/](src/): Main code files for analysis and output generation
   - [archive/](src/archive/): Contains old code used during testing
   - [preprocess.py](src/preprocess.py): Contains complete preprocessing pipeline used before analysis
   - [sentiment_analysis.py](src/sentiment_analysis.py): Contains code for sentiment analysis as a form of histograms and confusion matrices
   - [snowball_stopwords_extract.R](src/snowball_stopwords_extract.R): Contains _R_ code to extract the stopwords list
   - [experiment3.py](src/experiment3.py): Contains code for ablation studies and threshold analysis.
   - [word_cloud.py](src/word_cloud.py): Contains code for word clouds and top 10 word counts plots
4. [LICENSE](LICENSE): Licence for the project
5. [README.md](README.md): Documentation for the project
6. [requirements.txt](requirements.txt): Python dependencies

## Setup

1. Since [data_stopwords_snowball.rda](https://github.com/quanteda/stopwords/blob/master/data/data_stopwords_snowball.rda) is an _R_ object, we need to extract the stopwords list for using it in _Python_.

   - Download [data_stopwords_snowball.rda](https://github.com/quanteda/stopwords/blob/master/data/data_stopwords_snowball.rda) document and keep in `assets` folder.

   - In _R Studio_, run the `src/snowball_stopwords_extract.R` file to get `stopwords_en.txt`. This is the `.txt` version of the snowball stopwords list in English language.

   - This list still contains some negators which are not ideal for sentiment analysis. Hence, we choose to remove: `["isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't","hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't", "against", "no", "nor", "not"]` words from the text file manually. One should go ahead and delete them as well.

   - Also, this stopwords list doesn't contain the token `'s` which is usually present in other stopwords list. Hence, we add `'s` token to the stopwords list as well.

2. The parameters needed for the code globally are saved in `params.json`.

   - `"data"`: contains parameters needed for extracting values

     - `"all_memories_path"`: contains the path to the excel file containing the dataset

     - `"stopwords_path"`: contains the path to the text file containing modified snowball stopwords list

     - `"sentiment_threshold"`: contains the threshold set for 3 bin sentiment classification between VADER and TextBlob for neutral statements

     - `"preprocess"`: contains `true` if preprocessing needs to be done before analysis, else `false`

     - `"sentiment_type"`: contains `"hist"` if sentiment analysis graphs need to be produced, else `"cm"` if sentiment analysis metrics and confusion matrices need to be produced

   - `"output"`: contains paths to all the output files; `wordcloud`, `wordcounts`, `sentiment histogram`, `confusion matrices` in the `output` folder.

3. Run the scripts in the [src/](src/) folder for specific tasks.

## Preprocessing Pipeline

The `preprocessing_pipeline` function in [src/preprocess.py](src/preprocess.py) file, and include the following steps:

1. **Tokenization**: Splits text into individual tokens.
2. **Lowercasing**: Converts all text to lowercase.
3. **Lemmatization**: Reduces words to their base forms.
4. **Stopword Removal**: Removing snowball stopwords (without negators and including 's token).
5. **Punctuation Removal**: Removes all punctuation marks.
6. **Custom Stopwords**: Finding custom stopwords based on words occuring >3SDs away and then removing those custom stopwords.

## Sentiment Analysis

The [src/sentiment_analysis.py](src/sentiment_analysis.py) script performs sentiment analysis for the two methods, VADER and TextBlob. It includes:

1. **VADER Sentiment Analysis**: Calculates sentiment scores using the VADER library.
2. **TextBlob Sentiment Analysis**: Calculates sentiment scores using the TextBlob library.
3. **Histogram Generation**: Generates histograms for sentiment scores.
4. **Confusion Matrices**: Compares predicted sentiment scores with ground truth values.

**NOTE**: One may need to uncomment/comment some lines when using _rIAMs_ dataset mentioned in 651 and 662 lines.

## Word Cloud and Word Counts

The word cloud and top 10 word count analysis are implemented in [src/word_cloud.py](src/word_cloud.py). It includes:

1. **Word Cloud Generation**: Creates word clouds for the most words in the dataset.
2. **Top 10 Word Counts**: Generates bar plots for the top 10 most frequent words.

## Ablation and Threshold Analysis

The ablation and optimal threshold analysis are implemented in [src/experiment3.py](src/experiment3.py). It includes:
src/experiment3.py. It includes:

1. **Ablation Studies**: Evaluates the performance of sentiment analysis models on subsets of the dataset.
2. **Optimal Threshold Analysis**:
   - 3-Class Threshold Analysis: Finds the best positive and negative thresholds for - 3-class sentiment classification.
   - 5-Class Threshold Analysis: Finds the best lower and upper bounds for 5-class sentiment classification.
3. **Metrics**: Calculates Matthews Correlation Coefficient (MCC) for both VADER and TextBlob.
4. **Visualization**:
   - Threshold analysis: Generates heatmaps for MCC scores across different thresholds. Saves reports summarizing the best thresholds and their corresponding MCC scores.
   - Ablation analysis: Generates lineplot for average MCC scores across different sample sizes. Saves reports summarizing the average MCC scoress across different datasets.

**NOTE**: One may need to make some changes at line 394 when using _rIAMs_ dataset.

## Running the Analysis

1. Set up the parameters in `params.json`
2. Run the preprocessing pipeline:
   ```bash
   python src/preprocess.py
   ```
   It is also integrated with the other scripts for use, so one may not need to run this separately.
3. Perform sentiment analysis and confusion matrices:
   ```bash
   python src/sentiment_analysis.py
   ```
4. Generate word cloud and top 10 word count visualizations:
   ```bash
   python src/word_cloud.py
   ```
5. Run ablation studies and optimal threshold analysis:
   ```bash
   python src/experiment3.py
   ```
