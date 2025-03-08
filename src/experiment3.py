import numpy as np
import pandas as pd
from sentiment_analysis import vader_score, textblob_score, polarity_score
from preprocess import preprocessing_pipeline
import json
from sklearn import metrics
import matplotlib.pyplot as plt

# Helper Function to create a subset of the data


def mySubset(ds, size, seed):

    np.random.seed(seed)
    idx = np.random.choice(ds.shape[0], size)
    return ds.iloc[idx, :]


def sentimentScores(df: pd.DataFrame, sentiment_threshold: float):
    '''
    df: The entire data frame for a given dataset
    sentiment_output_path: Output path for the plots
    The function returns the saves the histogram for the sentiment analysis in all of dataset.
    '''

    # Extract the 'Memory_Text' column
    # Drop NaN values and convert to a list
    # cnt = len(memory_texts)
    memory_texts = df['Memory_text'].dropna().tolist()
    selfvalence_scores_5 = df['Valence'].dropna().tolist()

    # self valence scores in 3-class classification
    selfvalence_scores_3 = [-1 if x in [1, 2]
                            else 0 if x == 3 else 1 for x in selfvalence_scores_5]

    cnt = len(memory_texts)

    vader_scores_5 = []
    textblob_scores_5 = []
    vader_scores_3 = []
    textblob_scores_3 = []
    vader_scores_raw = []
    textblob_scores_raw = []

    for memory in memory_texts:
        _, _, _, vader = vader_score(memory)
        txtblob, _ = textblob_score(memory)
        vader_scores_5.append(polarity_score(
            vader, sentiment_threshold, vis=1))
        textblob_scores_5.append(polarity_score(
            txtblob, sentiment_threshold, vis=1))
        vader_scores_3.append(polarity_score(
            vader, sentiment_threshold, vis=3))
        textblob_scores_3.append(polarity_score(
            txtblob, sentiment_threshold, vis=3))
        vader_scores_raw.append(vader)
        textblob_scores_raw.append(txtblob)

    return vader_scores_5, textblob_scores_5, vader_scores_3, textblob_scores_3, selfvalence_scores_5, selfvalence_scores_3, vader_scores_raw, textblob_scores_raw


def mcc_scores(vader_scores, textblob_scores, vader_scores_vis3, textblob_scores_vis3, selfvalence_scores, selfvalence_scores_vis3):

    mcc_vader = metrics.matthews_corrcoef(
        vader_scores, selfvalence_scores)
    mcc_textblob = metrics.matthews_corrcoef(
        textblob_scores, selfvalence_scores)
    mcc_vader_vis3 = metrics.matthews_corrcoef(
        vader_scores_vis3, selfvalence_scores_vis3)
    mcc_textblob_vis3 = metrics.matthews_corrcoef(
        textblob_scores_vis3, selfvalence_scores_vis3)

    return mcc_vader, mcc_textblob, mcc_vader_vis3, mcc_textblob_vis3


def averageMCCs(df: pd.DataFrame, size: int, samples: int, percentage: int = None):
    '''
    df: The entire data frame for a given dataset
    size: The size of the subset
    samples: The number of samples to take for each subset size
    '''
    tbMCCs_3c = []
    vMCCs_3c = []
    tbMCCs_5c = []
    vMCCs_5c = []
    print(
        f'\nCalculating Average MCC for {percentage}% subset size: {size}, for samples: {samples}')
    for i in range(samples):
        # print(i)
        subset = mySubset(df, size, i)
        # vader_scores_5, textblob_scores_5, vader_scores_3, textblob_scores_3, selfvalence_scores_5, selfvalence_scores_3 = sentimentScores(
        #     subset)
        vader_scores_5, textblob_scores_5, vader_scores_3, textblob_scores_3, selfvalence_scores_5, selfvalence_scores_3 = subset['Vader_5c'].tolist(
        ), subset['TextBlob_5c'].tolist(), subset['Vader_3c'].tolist(), subset['TextBlob_3c'].tolist(), subset['Selfvalence_5c'].tolist(), subset['Selfvalence_3c'].tolist()
        mcc_vader_5, mcc_textblob_5, mcc_vader_3, mcc_textblob_3 = mcc_scores(
            vader_scores_5, textblob_scores_5, vader_scores_3, textblob_scores_3, selfvalence_scores_5, selfvalence_scores_3)
        tbMCCs_3c.append(mcc_textblob_3)
        vMCCs_3c.append(mcc_vader_3)
        tbMCCs_5c.append(mcc_textblob_5)
        vMCCs_5c.append(mcc_vader_5)

    out = pd.DataFrame({"TextBlobMCCs_3c": tbMCCs_3c,
                        "VaderMCCs_3c": vMCCs_3c,
                        "TextBlobMCCs_5c": tbMCCs_5c,
                        "VaderMCCs_5c": vMCCs_5c})

    print("Average 3-Class MCC - TextBlob: ", np.mean(out["TextBlobMCCs_3c"]))
    print("Average 3-Class MCC - Vader: ", np.mean(out["VaderMCCs_3c"]))
    print("Average 5-Class MCC - TextBlob: ", np.mean(out["TextBlobMCCs_5c"]))
    print("Average 5-Class MCC - Vader: ", np.mean(out["VaderMCCs_5c"]))

    return out


def write_ablation_report(out: pd.DataFrame, size: int, samples: int, percentage: int = None):
    text = f'''
    Calculating Average MCC for {percentage}% subset size: {size}, for samples: {samples}
    Average 3-Class MCC - TextBlob: {np.mean(out["TextBlobMCCs_3c"])}
    Average 3-Class MCC - Vader: {np.mean(out["VaderMCCs_3c"])}
    Average 5-Class MCC - TextBlob: {np.mean(out["TextBlobMCCs_5c"])}
    Average 5-Class MCC - Vader: {np.mean(out["VaderMCCs_5c"])}

    '''
    return text


def save_ablation_report_plot(df: pd.DataFrame, datatype: str, out100: pd.DataFrame, samples: int = 50):
    '''
    It saves all the plots and the reports in a text file
    '''

    if datatype != 'rIAMS':
        (out75, out50, out33, out20, out10) = (averageMCCs(
            df=df, size=615, samples=samples, percentage=75),
            averageMCCs(df=df, size=410, samples=samples, percentage=50),
            averageMCCs(df=df, size=275, samples=samples, percentage=33),
            averageMCCs(df=df, size=160, samples=samples, percentage=20),
            averageMCCs(df=df, size=82, samples=samples, percentage=10))
        sample_sizes = [825, 615, 410, 275, 160, 82]
    else:
        (out75, out50, out33, out20, out10) = (averageMCCs(
            df=df, size=1800, samples=samples, percentage=75),
            averageMCCs(df=df, size=1200, samples=samples, percentage=50),
            averageMCCs(df=df, size=825, samples=samples, percentage=33),
            averageMCCs(df=df, size=500, samples=samples, percentage=20),
            averageMCCs(df=df, size=250, samples=samples, percentage=10))
        sample_sizes = [2484, 1800, 1200, 825, 500, 250]

    mean_tb_mccs_5c = [np.mean(out100['TextBlobMCCs_5c']), np.mean(out75['TextBlobMCCs_5c']),
                       np.mean(out50['TextBlobMCCs_5c']), np.mean(
                           out33['TextBlobMCCs_5c']),
                       np.mean(out20['TextBlobMCCs_5c']), np.mean(out10['TextBlobMCCs_5c'])]
    mean_v_mccs_5c = [np.mean(out100['VaderMCCs_5c']), np.mean(out75['VaderMCCs_5c']),
                      np.mean(out50['VaderMCCs_5c']), np.mean(
                          out33['VaderMCCs_5c']),
                      np.mean(out20['VaderMCCs_5c']), np.mean(out10['VaderMCCs_5c'])]
    mean_tb_mccs_3c = [np.mean(out100['TextBlobMCCs_3c']), np.mean(out75['TextBlobMCCs_3c']),
                       np.mean(out50['TextBlobMCCs_3c']), np.mean(
                           out33['TextBlobMCCs_3c']),
                       np.mean(out20['TextBlobMCCs_3c']), np.mean(out10['TextBlobMCCs_3c'])]
    mean_v_mccs_3c = [np.mean(out100['VaderMCCs_3c']), np.mean(out75['VaderMCCs_3c']),
                      np.mean(out50['VaderMCCs_3c']), np.mean(
                          out33['VaderMCCs_3c']),
                      np.mean(out20['VaderMCCs_3c']), np.mean(out10['VaderMCCs_3c'])]

    plt.plot(sample_sizes, mean_v_mccs_3c, color='lightcoral',
             marker='o', label="3-Class: Vader")
    plt.plot(sample_sizes, mean_v_mccs_5c, color='lightcoral',
             linestyle='dashed', marker='o', label="5-Class: Vader")
    plt.plot(sample_sizes, mean_tb_mccs_3c, color='darkturquoise',
             marker='o', label="3-Class: TextBlob")
    if mean_tb_mccs_5c[0] < 0:
        plt.plot(sample_sizes, np.array(mean_tb_mccs_5c)*-1, color='darkturquoise',
                 linestyle='dashed', marker='o', label="5-Class: TextBlob")
    else:
        plt.plot(sample_sizes, mean_tb_mccs_5c, color='darkturquoise',
                 linestyle='dashed', marker='o', label="5-Class: TextBlob")
    plt.xlabel("Sample Size")
    plt.ylabel("Average MCC")
    plt.ylim([0, 0.5])
    plt.legend()
    plt.savefig(f'output/ablation_{datatype}.png')
    plt.close()
    with open(f'output/ablation_{datatype}.txt', 'w') as f:
        f.write(
            f'Ablation MCC results for {datatype} Report\n')
        if datatype != 'rIAMS':
            f.write(write_ablation_report(out100, 825, samples, 100))
            f.write(write_ablation_report(out75, 615, samples, 75))
            f.write(write_ablation_report(out50, 410, samples, 50))
            f.write(write_ablation_report(out33, 275, samples, 33))
            f.write(write_ablation_report(out20, 160, samples, 20))
            f.write(write_ablation_report(out10, 82, samples, 10))
        else:
            f.write(write_ablation_report(out100, 2484, samples, 100))
            f.write(write_ablation_report(out75, 1800, samples, 75))
            f.write(write_ablation_report(out50, 1200, samples, 50))
            f.write(write_ablation_report(out33, 825, samples, 33))
            f.write(write_ablation_report(out20, 500, samples, 20))
            f.write(write_ablation_report(out10, 250, samples, 10))
    f.close()
    return out100, out75, out50, out33, out20, out10, mean_tb_mccs_3c, mean_v_mccs_3c, mean_tb_mccs_5c, mean_v_mccs_5c, sample_sizes


def asymmetric_3_polarity(a: list, n: float, p: float):
    polarities = []
    for x in a:
        if x > p:
            polarities.append(1)
        elif x < n:
            polarities.append(-1)
        else:
            polarities.append(0)
    return polarities

# 3 Class Threshold Analysis function


def threeClassTA(df: pd.DataFrame):
    tbScores = df["TextBlob_raw"]
    vScores = df["Vader_raw"]
    real3 = df["Selfvalence_3c"]

    positiveThresholds = np.arange(0.00, 1, 0.01)
    negativeThresholds = np.arange(0.00, 1, 0.01) * -1

    tbMCC = []
    vMCC = []
    pT = []
    nT = []
    for p in positiveThresholds:
        for n in negativeThresholds:
            tbScored = asymmetric_3_polarity(tbScores, n, p)
            vScored = asymmetric_3_polarity(vScores, n, p)

            pT.append(p)
            nT.append(n)

            tbMCC.append(metrics.matthews_corrcoef(real3, tbScored))
            vMCC.append(metrics.matthews_corrcoef(real3, vScored))

    return pd.DataFrame({'PositiveThresholds': pT, 'NegativeThresholds': nT, 'TextBlobMCC': tbMCC, 'VADERMCC': vMCC})

# Returns a 5 class score


def fiveClass_score(score: float, lower: float, upper: float):
    '''
    Individual score to 5 class score
    '''
    if score > upper:
        return 2
    elif score > lower:
        return 1
    elif score >= -1*lower:
        return 0
    elif score >= -1*upper:
        return -1
    elif score >= -1.0:
        return -2

# Wrapper function that applies fiveClass_score to an array


def fcScore_list(a: list, lb: float, ub: float):
    out = []
    for x in a:
        out.append(fiveClass_score(x, lb, ub))
    return out

# 5 Class Threhsold Analysis Function


def fiveClassTA(df: pd.DataFrame):
    tbScores = df["TextBlob_raw"]
    vScores = df["Vader_raw"]
    real5 = df["Selfvalence_5c"]

    real5 = [-2 if x == 1 else -1 if x == 2 else 0 if x ==
             3 else 1 if x == 4 else 2 for x in real5]

    lowerbounds = np.arange(0.00, 1, 0.01)

    tbMCC = []
    vMCC = []
    lbT = []
    ubT = []
    for lb in lowerbounds:
        upperbounds = np.arange(lb+0.01, 1, 0.01)
        for ub in upperbounds:
            # print('ho')
            tbScored = fcScore_list(tbScores, lb, ub)
            vScored = fcScore_list(vScores, lb, ub)

            lbT.append(lb)
            ubT.append(ub)

            tbMCC.append(metrics.matthews_corrcoef(real5, tbScored))
            vMCC.append(metrics.matthews_corrcoef(real5, vScored))

    return pd.DataFrame({'Lowerbounds': lbT, 'Upperbounds': ubT, 'TextBlobMCC': tbMCC, 'VADERMCC': vMCC})


def heatPlots(df3: pd.DataFrame, df5: pd.DataFrame, datatype: str = 'MEAMS'):
    # Makes the MCC heat plots
    # df3 is the 3classTA ds
    # df5 is the 5classTA ds

    plt.subplot(2, 2, 1)
    plt.scatter(df3["PositiveThresholds"], df3["NegativeThresholds"],
                c=df3["VADERMCC"], s=2, linewidth=2, cmap="coolwarm")
    clb = plt.colorbar()
    clb.ax.set_title(' MCC', size=7)
    plt.title("Vader MCC Scores")
    plt.xlabel("Positive Threshold")
    plt.ylabel("Negative Threshold")

    plt.subplot(2, 2, 2)
    plt.scatter(df3["PositiveThresholds"], df3["NegativeThresholds"],
                c=df3["TextBlobMCC"], s=2, linewidth=2, cmap="coolwarm")
    clb = plt.colorbar()
    clb.ax.set_title(' MCC', size=7)
    plt.title("TextBlob MCC Scores")
    plt.xlabel("Positive Threshold")

    plt.subplot(2, 2, 3)
    plt.scatter(df5["Lowerbounds"], df5["Upperbounds"],
                c=df5["VADERMCC"], s=2, linewidth=2, cmap="coolwarm")
    clb = plt.colorbar()
    clb.ax.set_title(' MCC', size=7)
    plt.xlabel("Lowerbounds")
    plt.ylabel("Upperbounds")

    plt.subplot(2, 2, 4)
    plt.scatter(df5["Lowerbounds"], df5["Upperbounds"],
                c=df5["TextBlobMCC"], s=2, linewidth=2, cmap="coolwarm")
    clb = plt.colorbar()
    clb.ax.set_title(' MCC', size=7)
    plt.xlabel("Lowerbounds")
    plt.subplots_adjust(hspace=.3, wspace=.3)

    plt.savefig(f'output/threshold_analysis_heatmaps_{datatype}.png')
    plt.close()
    return


def save_thresholf_report_plot(out3: pd.DataFrame, out5: pd.DataFrame, datatype: str = 'MEAMS'):
    '''
    It saves all the plots and the reports in a text file
    '''

    heatPlots(out3, out5, datatype=datatype)
    with open(f'output/threshold_analysis_report_{datatype}.txt', 'w') as f:
        text = f'''
        Threshold Analysis and MCC results for {datatype} Report:\n
        Best 3-Class Thresholds - VADER: Positive: {np.round(out3["PositiveThresholds"][out3["VADERMCC"].argmax()], 2)}, Negative: {np.round(out3["NegativeThresholds"][out3["VADERMCC"].argmax()], 2)}
        Best 3-Class Thresholds - TextBlob: Positive: {np.round(out3["PositiveThresholds"][out3["TextBlobMCC"].argmax()], 2)}, Negative: {np.round(out3["NegativeThresholds"][out3["TextBlobMCC"].argmax()], 2)}
        Best 5-Class Thresholds - Vader: Lowerbound: {np.round(out5["Lowerbounds"][out5["VADERMCC"].argmax()], 2)}, Upperbound: {np.round(out5["Upperbounds"][out5["VADERMCC"].argmax()], 2)}
        Best 5-Class Thresholds - TextBlob: Lowerbound: {np.round(out5["Lowerbounds"][out5["TextBlobMCC"].argmax()], 2)}, Upperbound: {np.round(out5["Upperbounds"][out5["TextBlobMCC"].argmax()], 2)}
        MCC of Best 3-Class Thresholds - VADER: {out3["VADERMCC"][out3["VADERMCC"].argmax()]}
        MCC of Best 3-Class Thresholds - TextBlob: {out3["TextBlobMCC"][out3["TextBlobMCC"].argmax()]}
        MCC of Best 5-Class Thresholds - VADER: {out5["VADERMCC"][out5["VADERMCC"].argmax()]}
        MCC of Best 5-Class Thresholds - TextBlob: {out5["TextBlobMCC"][out5["TextBlobMCC"].argmax()]}
        '''
        f.write(text)
    f.close()
    return


if __name__ == '__main__':

    np.random.seed(1234)

    # Open the parameter file to get necessary parameters
    param_filename = 'src/params.json'
    with open(param_filename) as paramfile:
        param = json.load(paramfile)

    # Load parameter values
    data_file_path = param['data']['all_memories_path']
    stopwords_path = param['data']['stopwords_path']
    preprocess = param['data']['preprocess']
    sentiment_threshold = param['data']['sentiment_threshold']

    # Read the stopwords txt file
    with open(stopwords_path, 'r') as file:
        stopwords = file.read().splitlines()

    # Read the excel file for the data
    df = pd.read_excel(data_file_path)
    df_columns = df.columns.to_list()
    print(f'Data columns are :: \n{df_columns}')

    # Mention the dataset being used MEAMS or rIAMS
    datatype = 'MEAMS'
    # datatype = 'rIAMS'

    if datatype == 'rIAMS':
        df['Memory_text'] = df['r_mem_s_4_text']
        df['Valence'] = df['r_mem_s_13_valence']
        df['Valence'] = df['Valence'].replace({-2: 1, -1: 2, 0: 3, 1: 4, 2: 5})

    # Preprocess the memory texts in the dataframe df
    if preprocess:
        memories_list = df['Memory_text'].to_list()
        memories_list = preprocessing_pipeline(memories_list, stopwords)
        df['Memory_text'] = memories_list

    vader_scores_5, textblob_scores_5, vader_scores_3, textblob_scores_3, selfvalence_scores_5, selfvalence_scores_3, vader_scores_raw, textblob_scores_raw = sentimentScores(
        df=df, sentiment_threshold=sentiment_threshold)
    mcc_vader_5, mcc_textblob_5, mcc_vader_3, mcc_textblob_3 = mcc_scores(
        vader_scores_5, textblob_scores_5, vader_scores_3, textblob_scores_3, selfvalence_scores_5, selfvalence_scores_3)

    df['Vader_3c'] = vader_scores_3
    df['TextBlob_3c'] = textblob_scores_3
    df['Vader_5c'] = vader_scores_5
    df['TextBlob_5c'] = textblob_scores_5
    df['Selfvalence_3c'] = selfvalence_scores_3
    df['Selfvalence_5c'] = selfvalence_scores_5
    df['Vader_raw'] = vader_scores_raw
    df['TextBlob_raw'] = textblob_scores_raw

    print('\n')
    print(
        f'Matthews Correlation Coefficient for Vader (5-class): {mcc_vader_5}')
    print(
        f'Matthews Correlation Coefficient for Textblob (5-class): {mcc_textblob_5}')
    print(
        f'Matthews Correlation Coefficient for Vader (3-class): {mcc_vader_3}')
    print(
        f'Matthews Correlation Coefficient for Textblob (3-class): {mcc_textblob_3}\n')

    # Calculating MCC on 100% (825 samples) of the data for original thresholds
    out100 = pd.DataFrame({"TextBlobMCCs_3c": [mcc_textblob_3],
                           "VaderMCCs_3c": [mcc_vader_3],
                           "TextBlobMCCs_5c": [mcc_textblob_5],
                           "VaderMCCs_5c": [mcc_vader_5]})

    out100, out75, out50, out33, out20, out10, mean_tb_mccs_3c, mean_v_mccs_3c, mean_tb_mccs_5c, mean_v_mccs_5c, sample_sizes = save_ablation_report_plot(
        df=df, datatype=datatype, out100=out100, samples=50)

    out3 = threeClassTA(df)
    out5 = fiveClassTA(df)

    save_thresholf_report_plot(out3, out5, datatype)
