# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import random
from sklearn import metrics
from preprocess import preprocessing_pipeline
import os
import matplotlib.ticker as mticker


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


'''
def vader_selfvalence_compare(vader_scores: list, selfvalence_scores: list):
    count11, count12, count13, count14, count15 = 0, 0, 0, 0, 0
    count21, count22, count23, count24, count25 = 0, 0, 0, 0, 0
    count31, count32, count33, count34, count35 = 0, 0, 0, 0, 0
    count41, count42, count43, count44, count45 = 0, 0, 0, 0, 0
    count51, count52, count53, count54, count55 = 0, 0, 0, 0, 0

    for i in range(0, len(vader_scores)):
        if selfvalence_scores[i] == 1:
            if vader_scores[i] == 1:
                count11 += 1
            elif vader_scores[i] == 2:
                count12 += 1
            elif vader_scores[i] == 3:
                count13 += 1
            elif vader_scores[i] == 4:
                count14 += 1
            elif vader_scores[i] == 5:
                count15 += 1
        elif selfvalence_scores[i] == 2:
            if vader_scores[i] == 1:
                count21 += 1
            elif vader_scores[i] == 2:
                count22 += 1
            elif vader_scores[i] == 3:
                count23 += 1
            elif vader_scores[i] == 4:
                count24 += 1
            elif vader_scores[i] == 5:
                count25 += 1
        elif selfvalence_scores[i] == 3:
            if vader_scores[i] == 1:
                count31 += 1
            elif vader_scores[i] == 2:
                count32 += 1
            elif vader_scores[i] == 3:
                count33 += 1
            elif vader_scores[i] == 4:
                count34 += 1
            elif vader_scores[i] == 5:
                count35 += 1
        elif selfvalence_scores[i] == 4:
            if vader_scores[i] == 1:
                count41 += 1
            elif vader_scores[i] == 2:
                count42 += 1
            elif vader_scores[i] == 3:
                count43 += 1
            elif vader_scores[i] == 4:
                count44 += 1
            elif vader_scores[i] == 5:
                count45 += 1
        elif selfvalence_scores[i] == 5:
            if vader_scores[i] == 1:
                count51 += 1
            elif vader_scores[i] == 2:
                count52 += 1
            elif vader_scores[i] == 3:
                count53 += 1
            elif vader_scores[i] == 4:
                count54 += 1
            elif vader_scores[i] == 5:
                count55 += 1

    count1 = np.array([count11, count12, count13, count14, count15])
    count2 = np.array([count21, count22, count23, count24, count25])
    count3 = np.array([count31, count32, count33, count34, count35])
    count4 = np.array([count41, count42, count43, count44, count45])
    count5 = np.array([count51, count52, count53, count54, count55])

    return count1, count2, count3, count4, count5

'''

# Complex but shorter version of the above function


def vader_selfvalence_compare(vader_scores: list, selfvalence_scores: list):
    counts = {s: {v: 0 for v in [1, 2, 3, 4, 5]}
              for s in [1, 2, 3, 4, 5]}

    for s, v in zip(selfvalence_scores, vader_scores):
        if s in counts and v in counts[s]:
            counts[s][v] += 1

    return [np.array([counts[s][v] for v in [1, 2, 3, 4, 5]]) for s in [1, 2, 3, 4, 5]]


def sentiment_histogram_vis2(vader_scores: list, textblob_scores: list, selfvalence_scores: list, cnt: int, sentiment_output_path: str, cue_val=None, cue_type=None):
    '''
    vader_scores: Takes in polarity scores generated from Vader method
    textblob_scores: Takes in polarity scores generated from Textblob method
    cnt: Count of each memories of a particular type
    sentiment_output_path: Output path for the plots
    cue_type: None (by default) if the dataset is not separated by cue types. If it is separated, enter the cue_type.
    cue_val: None (by default) if the dataset is not separated by cue types. If it is separated, enter the individual cue_val from the cue_type.
    The function saves a histogram plot in the respective output location.
    '''
    # bins = np.arange(-1, 3)

    # print(selfvalence_scores)
    bins = np.arange(0.5, 6.5, step=1)
    # print(bins)

    # Creating the histogram data
    hist_vader, _ = np.histogram(vader_scores, bins=bins)
    # print(hist_vader)
    hist_textblob, _ = np.histogram(textblob_scores, bins=bins)
    # print(hist_textblob)
    hist_selfvalence, _ = np.histogram(selfvalence_scores, bins=bins)
    # print(hist_selfvalence)

    # Normalize histograms to get probabilities
    hist_vader_prob = hist_vader / sum(hist_vader)
    hist_textblob_prob = hist_textblob / sum(hist_textblob)
    hist_selfvalence_prob = hist_selfvalence / sum(hist_selfvalence)

    count_vader1, count_vader2, count_vader3, count_vader4, count_vader5 = vader_selfvalence_compare(
        vader_scores=vader_scores, selfvalence_scores=selfvalence_scores)
    count_vader1, count_vader2, count_vader3, count_vader4, count_vader5 = count_vader1 / sum(hist_selfvalence), count_vader2 / sum(
        hist_selfvalence), count_vader3 / sum(hist_selfvalence), count_vader4 / sum(hist_selfvalence), count_vader5 / sum(hist_selfvalence)
    count_textblob1, count_textblob2, count_textblob3, count_textblob4, count_textblob5 = vader_selfvalence_compare(
        vader_scores=textblob_scores, selfvalence_scores=selfvalence_scores)
    count_textblob1, count_textblob2, count_textblob3, count_textblob4, count_textblob5 = count_textblob1 / sum(hist_selfvalence), count_textblob2 / sum(
        hist_selfvalence), count_textblob3 / sum(hist_selfvalence), count_textblob4 / sum(hist_selfvalence), count_textblob5 / sum(hist_selfvalence)

    # Setting the width of each bar
    width = 0.5

    # Creating the figure and axes
    _, ax = plt.subplots()
    bins = np.arange(1, 6.5, step=1)
    # print(bins)

    ax.bar(bins[:-1] - width/4, count_vader1, width=width/2,
           label='Self valence 1', alpha=0.9, linewidth=1, color='k')
    ax.bar(bins[:-1] - width/4, count_vader2, width=width/2,
           label='Self valence 2', alpha=0.9, linewidth=1, bottom=count_vader1, color='grey')
    ax.bar(bins[:-1] - width/4, count_vader3, width=width/2,
           label='Self valence 3', alpha=0.9, linewidth=1, bottom=count_vader1+count_vader2, color='darkgrey')
    ax.bar(bins[:-1] - width/4, count_vader4, width=width/2,
           label='Self valence 4', alpha=0.9, linewidth=1, bottom=count_vader1+count_vader2+count_vader3, color='lightgrey')
    ax.bar(bins[:-1] - width/4, count_vader5, width=width/2,
           label='Self valence 5', alpha=0.9, linewidth=1, bottom=count_vader1+count_vader2+count_vader3+count_vader4, color='whitesmoke')
    ax.bar(bins[:-1] + width/4, count_textblob1, width=width/2,
           alpha=0.9, linewidth=1, color='k')
    ax.bar(bins[:-1] + width/4, count_textblob2, width=width/2,
           alpha=0.9, linewidth=1, bottom=count_textblob1, color='grey')
    ax.bar(bins[:-1] + width/4, count_textblob3, width=width/2,
           alpha=0.9, linewidth=1, bottom=count_textblob1+count_textblob2, color='darkgrey')
    ax.bar(bins[:-1] + width/4, count_textblob4, width=width/2,
           alpha=0.9, linewidth=1, bottom=count_textblob1+count_textblob2+count_textblob3, color='lightgrey')
    ax.bar(bins[:-1] + width/4, count_textblob5, width=width/2,
           alpha=0.9, linewidth=1, bottom=count_textblob1+count_textblob2+count_textblob3+count_textblob4, color='whitesmoke')

    ax.bar(bins[:-1] - width/4, hist_vader_prob, width=width/2,
           label='VADER analysis', alpha=0.75, linewidth=1.5, fill=False, edgecolor='blue')
    ax.bar(bins[:-1] + width/4, hist_textblob_prob, width=width/2,
           label='TextBlob analysis', alpha=0.75, linewidth=1.5, fill=False, edgecolor='red')

    # Adding titles and labels
    ax.set_xlabel('Valence scores')
    ax.set_ylabel('Proportion of Classification')

    # Set x-ticks to bin values
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels([f'{round(b, 1)}' for b in bins[:-1]])
    ax.legend()
    plt.xlim(0.5, 5.5)

    # Saving the plot
    if cue_type != None:
        plt.title(
            f'Cue type: {cue_type}, cue value: {cue_val}, # memories: {cnt}')
        plt.tight_layout()
        plt.savefig(
            f'{sentiment_output_path}/{cue_type}/{cue_val}_{cnt}_vis2.png', format='PNG')
    else:
        plt.title(f'All Memories, # memories: {cnt}')
        plt.tight_layout()
        plt.savefig(
            f'{sentiment_output_path}/All_Memories_{cnt}_vis2.png', format='PNG')

    # plt.show()
    plt.close()


def sentiment_histogram_vis1(vader_scores: list, textblob_scores: list, selfvalence_scores: list, cnt: int, sentiment_output_path: str, cue_val=None, cue_type=None):
    '''
    vader_scores: Takes in polarity scores generated from Vader method
    textblob_scores: Takes in polarity scores generated from Textblob method
    cnt: Count of each memories of a particular type
    sentiment_output_path: Output path for the plots
    cue_type: None (by default) if the dataset is not separated by cue types. If it is separated, enter the cue_type.
    cue_val: None (by default) if the dataset is not separated by cue types. If it is separated, enter the individual cue_val from the cue_type.
    The function saves a histogram plot in the respective output location.
    '''
    # bins = np.arange(-1, 3)

    display_labels = ["Very\nNegative", "Somewhat\nNegative",
                      "Neutral", "Somewhat\nPositive", "Very\nPositive"]

    # print(selfvalence_scores)
    bins = np.arange(0.5, 6.5, step=1)
    # print(bins)

    # Creating the histogram data
    hist_vader, _ = np.histogram(vader_scores, bins=bins)
    # print(hist_vader)
    hist_textblob, _ = np.histogram(textblob_scores, bins=bins)
    # print(hist_textblob)
    hist_selfvalence, _ = np.histogram(selfvalence_scores, bins=bins)
    # print(hist_selfvalence)

    # Normalize histograms to get probabilities
    hist_vader_prob = (hist_vader / sum(hist_vader))*100
    hist_textblob_prob = (hist_textblob / sum(hist_textblob))*100
    hist_selfvalence_prob = (hist_selfvalence / sum(hist_selfvalence))*100

    # Setting the width of each bar
    width = 0.5

    # Creating the figure and axes
    _, ax = plt.subplots()

    # Plotting the histograms
    bins = np.arange(1, 6.5, step=1)
    # print(bins)
    ax.bar(bins[:-1], hist_selfvalence_prob, width=width,
           label='Self-Reported Valence', alpha=0.5, linewidth=1, color='grey', edgecolor='darkgrey')

    ax.bar(bins[:-1] - width/4, hist_vader_prob, width=width/2,
           label='VADER', alpha=0.75, linewidth=1.5, fill=False, edgecolor='blue')
    ax.bar(bins[:-1] + width/4, hist_textblob_prob, width=width/2,
           label='TextBlob', alpha=0.75, linewidth=1.5, fill=False, edgecolor='red')

    # Adding titles and labels
    ax.set_xlabel('Valence scores')
    ax.set_ylabel('Proportion of Classification')

    # Set x-ticks to bin values
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels([f'{round(b, 1)}' for b in bins[:-1]])
    ax.set_xticklabels(display_labels, ha="center")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    plt.xlim(0.5, 5.5)

    # Saving the plot
    if cue_type != None:
        # plt.title(
        #     f'Cue type: {cue_type}, cue value: {cue_val}, # memories: {cnt}')
        plt.tight_layout()
        plt.savefig(
            f'{sentiment_output_path}/{cue_type}/{cue_val}_{cnt}_vis1.png', format='PNG')
    else:
        # plt.title(f'All Memories, # memories: {cnt}')
        plt.tight_layout()
        plt.savefig(
            f'{sentiment_output_path}/All_Memories_{cnt}_vis1.png', format='PNG')

    # plt.show()
    plt.close()


def sentiment_histogram_vis3(vader_scores: list, textblob_scores: list, cnt: int, sentiment_output_path: str, cue_val=None, cue_type=None):
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
    _, ax = plt.subplots()

    # Plotting the histograms
    ax.bar(bins[:-1] - width/2, hist_vader / sum(hist_vader), width=width,
           label='VADER analysis', color='blue')
    ax.bar(bins[:-1] + width/2, hist_textblob / sum(hist_textblob), width=width,
           label='Textblob analysis', color='orange')

    # Adding titles and labels
    ax.set_xlabel('Valence scores')
    ax.set_ylabel('Proportion of Classification')
    ax.legend()
    plt.xlim(-1.5, 1.5)

    # Saving the plot
    if cue_type != None:
        ax.set_title(
            f'Cue type: {cue_type}, cue value: {cue_val}, # memories: {cnt}')
        plt.tight_layout()
        plt.savefig(
            f'{sentiment_output_path}/{cue_type}/{cue_val}_{cnt}_vis3.png', format='PNG')
    else:
        ax.set_title(f'All Memories, # memories: {cnt}')
        plt.tight_layout()
        plt.savefig(
            f'{sentiment_output_path}/All_Memories_{cnt}_vis3.png', format='PNG')

    # plt.show()
    # exit()
    plt.close()


def confusion_matrix(vader_scores: list, textblob_scores: list, selfvalence_scores: list, cnt: int, confusion_output_path: str, cue_val=None, cue_type=None):

    # print(vader_scores, textblob_scores, selfvalence_scores)
    labels = [1, 2, 3, 4, 5]
    display_labels = ["Very\nNegative", "Somewhat\nNegative",
                      "Neutral", "Somewhat\nPositive", "Very\nPositive"]

    # Check the unique values of the true and predicted (VADER, Textblob) values.
    '''    
    unique_values, counts = np.unique(
        np.array(vader_scores), return_counts=True)
    print('Vader scores: ', unique_values, counts)
    unique_values, counts = np.unique(
        np.array(textblob_scores), return_counts=True)
    print('Textblob scores: ', unique_values, counts)
    # print('Self valence scores: ')
    unique_values, counts = np.unique(
        np.array(selfvalence_scores), return_counts=True)
    print('Self valence scores: ', unique_values, counts, '\n')

    df_vader, df_textblob, df_self = pd.DataFrame([vader_scores]), pd.DataFrame(
        [textblob_scores]), pd.DataFrame([selfvalence_scores])
    print(df_vader.nunique(axis=1), df_textblob.nunique(axis=1),
          df_self.nunique(axis=1))
    '''
    cm_vader = metrics.confusion_matrix(
        np.array(selfvalence_scores), np.array(vader_scores), labels=labels)
    cm_textblob = metrics.confusion_matrix(
        np.array(selfvalence_scores), np.array(textblob_scores), labels=labels)

    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(6, 12))
    display_vader = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm_vader, display_labels=display_labels)
    display_textblob = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm_textblob, display_labels=display_labels)

    p1 = display_vader.plot(cmap='Blues', ax=ax[0], values_format='d')
    p2 = display_textblob.plot(cmap='Blues', ax=ax[1], values_format='d')

    # p1.im_.colorbar.set_label("Narrative Count")
    # p2.im_.colorbar.set_label("Narrative Count")

    # Move color bar label to the top
    for disp in [p1, p2]:  # Iterate through both confusion matrices
        cbar = disp.im_.colorbar  # Get color bar
        cbar.ax.set_ylabel("")  # Remove the side label
        # Move label to the top
        cbar.ax.set_xlabel("Narrative\nCount")
        cbar.ax.xaxis.set_label_position('top')  # Set label position to top

    ax[0].set_title('Confusion Matrix')
    ax[0].set_xlabel('VADER labels')
    ax[0].set_ylabel('Self-Reported labels')
    ax[1].set_title('Confusion Matrix')
    ax[1].set_xlabel('TextBlob labels')
    ax[1].set_ylabel('Self-Reported labels')
    # ax[0].set_xticklabels(display_labels, va="center", ha="center")
    # ax[1].set_xticklabels(display_labels, va="center", ha="center")
    # ax[0].set_yticklabels(display_labels, va="center", ha="center")
    # ax[1].set_yticklabels(display_labels, va="center", ha="center")

    if cue_type != None:
        # plt.suptitle(
        #     f'Cue type: {cue_type}, cue value: {cue_val}, # memories: {cnt}', fontsize=16)
        fig.tight_layout()
        plt.savefig(
            f'{confusion_output_path}/{cue_type}/{cue_val}_{cnt}.png', format='PNG')
    else:
        # plt.suptitle(f'All Memories, # memories: {cnt}', fontsize=16)
        fig.tight_layout()
        plt.savefig(
            f'{confusion_output_path}/All_Memories_{cnt}.png', format='PNG')

    # plt.show()
    plt.close()

    # Compute all the metrics now for VADER and TextBlob together

    accuracy_vader = np.trace(cm_vader) / float(np.sum(cm_vader))
    accuracy_textblob = np.trace(cm_textblob) / float(np.sum(cm_textblob))

    precision_vader = metrics.precision_score(
        np.array(selfvalence_scores), np.array(vader_scores), average=None, labels=labels, zero_division=0)
    recall_vader = metrics.recall_score(
        np.array(selfvalence_scores), np.array(vader_scores), average=None, labels=labels, zero_division=0)
    # specificity_vader = metrics.recall_score(
    #     np.array(selfvalence_scores), np.array(vader_scores), average=None, labels=[1, 2, 3, 4, 5], zero_division=0, pos_label=0)
    f1_vader = metrics.f1_score(
        np.array(selfvalence_scores), np.array(vader_scores), average=None, labels=labels, zero_division=0)

    precision_textblob = metrics.precision_score(
        np.array(selfvalence_scores), np.array(textblob_scores), average=None, labels=labels, zero_division=0)
    recall_textblob = metrics.recall_score(
        np.array(selfvalence_scores), np.array(textblob_scores), average=None, labels=labels, zero_division=0)
    # specificity_textblob = metrics.recall_score(
    #     np.array(selfvalence_scores), np.array(textblob_scores), pos_label=0, average=None, labels=[1, 2, 3, 4, 5], zero_division=0)
    f1_textblob = metrics.f1_score(
        np.array(selfvalence_scores), np.array(textblob_scores), average=None, labels=labels, zero_division=0)

    if cue_type != None:
        computed_report = f'\n\t\t VADER \t\t\t Textblob \nAccuracy: {accuracy_vader}, {accuracy_textblob} \nPrecision: {precision_vader}, {precision_textblob} \nRecall: {recall_vader}, {recall_textblob} \nF1: {f1_vader}, {f1_textblob}'
        vader_cls_report = f'\n\nVADER \t\n' + metrics.classification_report(y_true=selfvalence_scores,
                                                                             y_pred=vader_scores, labels=labels, target_names=['1', '2', '3', '4', '5'], zero_division=0)
        textblob_cls_report = f'\nTextblob \t\n' + metrics.classification_report(y_true=selfvalence_scores,
                                                                                 y_pred=textblob_scores, labels=labels, target_names=['1', '2', '3', '4', '5'], zero_division=0)
        with open(f'{confusion_output_path}/{cue_type}/{cue_val}_{cnt}.txt', 'w') as f:
            f.write(
                f'TextBlob Classification Report\t\t\t-> Cue type: {cue_type}, cue value: {cue_val}, # memories: {cnt}\n')
            f.write(computed_report)
            f.write(vader_cls_report)
            f.write(textblob_cls_report)
        f.close()
    else:
        computed_report = f'\n\t\t VADER \t\t\t Textblob \nAccuracy: {accuracy_vader}, {accuracy_textblob} \nPrecision: {precision_vader}, {precision_textblob} \nRecall: {recall_vader}, {recall_textblob} \nF1: {f1_vader}, {f1_textblob}'
        # print(selfvalence_scores, vader_scores)
        vader_cls_report = f'\n\nVADER \t\n' + metrics.classification_report(y_true=selfvalence_scores,
                                                                             y_pred=vader_scores, target_names=['1', '2', '3', '4', '5'], zero_division=0)
        textblob_cls_report = f'\nTextblob \t\n' + metrics.classification_report(y_true=selfvalence_scores,
                                                                                 y_pred=textblob_scores, target_names=['1', '2', '3', '4', '5'], zero_division=0)
        with open(f'{confusion_output_path}/All_Memories_{cnt}.txt', 'w') as f:
            f.write(
                f'TextBlob Classification Report\t\t\t-> All Memories, # memories: {cnt}\n')
            f.write(computed_report)
            f.write(vader_cls_report)
            f.write(textblob_cls_report)
        f.close()

    # print(computed_report, vader_cls_report, textblob_cls_report)


def polarity_score(score: float, sentiment_threshold: float, vis=1):
    if vis != 3:
        if score > 0.6:
            return 5
        elif score > 0.2:
            return 4
        elif score >= -0.2:
            return 3
        elif score >= -0.6:
            return 2
        elif score >= -1.0:
            return 1
    else:
        if score > sentiment_threshold:
            return 1
        elif score < (-1 * sentiment_threshold):
            return -1
        else:
            return 0


def sentimentByCueType(cue_type: str, df: pd.DataFrame, sentiment_output_path: str, confusion_output_path: str, method: str):
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
        # Extract the 'Memory_Text' column
        # Drop NaN values and convert to a list
        filtered_df = df[df[cue_type] == cue_val]
        memory_texts = filtered_df['Memory_text'].dropna().tolist()
        selfvalence_scores = filtered_df['Valence'].dropna().tolist()
        cnt = len(memory_texts)

        vader_scores = []
        textblob_scores = []
        vader_scores_vis3 = []
        textblob_scores_vis3 = []

        # Parse through all the memories
        for memory in memory_texts:
            # Compute VADER and Textblob score of individual memory
            _, _, _, vader = vader_score(memory)
            txtblob, _ = textblob_score(memory)

            # Make an array of the VADER and Textblob scores of the memories
            vader_scores.append(polarity_score(
                vader, sentiment_threshold, vis=1))
            textblob_scores.append(polarity_score(
                txtblob, sentiment_threshold, vis=1))
            vader_scores_vis3.append(polarity_score(
                vader, sentiment_threshold, vis=3))
            textblob_scores_vis3.append(polarity_score(
                txtblob, sentiment_threshold, vis=3))

        if method == 'hist':
            sentiment_histogram_vis1(vader_scores=vader_scores, textblob_scores=textblob_scores, selfvalence_scores=selfvalence_scores, cnt=cnt,
                                     sentiment_output_path=sentiment_output_path, cue_val=cue_val, cue_type=cue_type)
            sentiment_histogram_vis2(vader_scores=vader_scores, textblob_scores=textblob_scores, selfvalence_scores=selfvalence_scores, cnt=cnt,
                                     sentiment_output_path=sentiment_output_path, cue_val=cue_val, cue_type=cue_type)
            sentiment_histogram_vis3(vader_scores=vader_scores_vis3, textblob_scores=textblob_scores_vis3, cnt=cnt,
                                     sentiment_output_path=sentiment_output_path, cue_val=cue_val, cue_type=cue_type)
        elif method == 'cm':
            confusion_matrix(vader_scores=vader_scores, textblob_scores=textblob_scores, selfvalence_scores=selfvalence_scores, cnt=cnt,
                             confusion_output_path=confusion_output_path, cue_val=cue_val, cue_type=cue_type)


def sentimentOverall(df: pd.DataFrame, sentiment_output_path: str, confusion_output_path: str, method: str):
    '''
    df: The entire data frame for a given dataset
    sentiment_output_path: Output path for the plots
    The function returns the saves the histogram for the sentiment analysis in all of dataset.
    '''

    # Extract the 'Memory_Text' column
    # Drop NaN values and convert to a list
    # cnt = len(memory_texts)
    memory_texts = df['Memory_text'].dropna().tolist()
    selfvalence_scores = df['Valence'].dropna().tolist()

    cnt = len(memory_texts)

    vader_scores = []
    textblob_scores = []
    vader_scores_vis3 = []
    textblob_scores_vis3 = []

    for memory in memory_texts:
        _, _, _, vader = vader_score(memory)
        txtblob, _ = textblob_score(memory)
        vader_scores.append(polarity_score(vader, sentiment_threshold, vis=1))
        textblob_scores.append(polarity_score(
            txtblob, sentiment_threshold, vis=1))
        vader_scores_vis3.append(polarity_score(
            vader, sentiment_threshold, vis=3))
        textblob_scores_vis3.append(polarity_score(
            txtblob, sentiment_threshold, vis=3))

    if method == 'hist':
        sentiment_histogram_vis1(vader_scores=vader_scores, textblob_scores=textblob_scores, selfvalence_scores=selfvalence_scores,
                                 cnt=cnt, sentiment_output_path=sentiment_output_path)
        sentiment_histogram_vis2(vader_scores=vader_scores, textblob_scores=textblob_scores, selfvalence_scores=selfvalence_scores,
                                 cnt=cnt, sentiment_output_path=sentiment_output_path)
        sentiment_histogram_vis3(vader_scores=vader_scores_vis3, textblob_scores=textblob_scores_vis3,
                                 cnt=cnt, sentiment_output_path=sentiment_output_path)
    elif method == 'cm':
        confusion_matrix(vader_scores=vader_scores, textblob_scores=textblob_scores,
                         selfvalence_scores=selfvalence_scores, cnt=cnt, confusion_output_path=confusion_output_path)


def check_create_folders(path):
    # Check if output folders exist or not, if not then create the folders
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':

    np.random.seed(1234)

    # Open the parameter file to get necessary parameters
    param_filename = 'src/params.json'
    with open(param_filename) as paramfile:
        param = json.load(paramfile)

    # Load parameter values
    seed_value = param['data']['seed_value']
    data_file_path = param['data']['all_memories_path']
    stopwords_path = param['data']['stopwords_path']
    preprocess = param['data']['preprocess']
    sentiment_threshold = param['data']['sentiment_threshold']
    sentiment_type = param['data']['sentiment_type']
    sentiment_output_path = param['output']['sentiment_output_path']
    confusion_output_path = param['output']['confusion_output_path']

    # Check if output folders exist or not, if not then create the folders
    check_create_folders(sentiment_output_path)
    check_create_folders(confusion_output_path)

    # Read the stopwords txt file
    with open(stopwords_path, 'r') as file:
        stopwords = file.read().splitlines()

    # Read the excel file for the data
    df = pd.read_excel(data_file_path)
    df_columns = df.columns.to_list()
    print(f'Data columns are :: \n{df_columns}')

    # Uncomment the below 3 lines for sentiment analysis of rIAMs dataset
    # df['Memory_text'] = df['r_mem_s_4_text']
    # df['Valence'] = df['r_mem_s_13_valence']
    # df['Valence'] = df['Valence'].replace({-2: 1, -1: 2, 0: 3, 1: 4, 2: 5})

    # Preprocess the memory texts in the dataframe df
    if preprocess:
        memories_list = df['Memory_text'].to_list()
        memories_list = preprocessing_pipeline(memories_list, stopwords)
        df['Memory_text'] = memories_list

    # Comment the below 8 lines for sentiment analysis of rIAMs dataset
    for cue_type in ['Song', 'Singer', 'Year', 'Condition']:
        # Check if output folders exist or not, if not then create the folders
        check_create_folders(path=f"{sentiment_output_path}/{cue_type}")
        check_create_folders(path=f"{confusion_output_path}/{cue_type}")
        # for cue_type in ['Year']:
        print(cue_type)
        sentimentByCueType(cue_type=cue_type, df=df,
                           sentiment_output_path=sentiment_output_path, confusion_output_path=confusion_output_path, method=sentiment_type)

    sentimentOverall(
        df=df, sentiment_output_path=sentiment_output_path, confusion_output_path=confusion_output_path, method=sentiment_type)
