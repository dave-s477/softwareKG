import pandas as pd

from os.path import join

from .eval_fct import evaluate_flat

def base_scores(result, save_location=''):
    """
    Calculate the basic scores given two sequences.
    Scoring is done in different way regarding overlap and entities.
    
    Arguments:
        result {[type]} -- [to sequences, truth and prediction]
    
    Keyword Arguments:
        save {bool} -- [save scores to file or just write output] (default: {True})
    
    Returns:
        [pd dataframe] -- [containing all scores]
    """
    df = pd.DataFrame()
    for key in result[1]:
        run = key.split('_')[-1].split('.')[0]
        print("Results for cross-validation fold {}:".format(run))
        gold_labels = result[1][key]
        predictions = result[0][key]
        res = evaluate_flat(gold_labels, predictions)
        res['fold'] = run
        df = df.append(res)
    if save_location:
        df.to_csv(join(save_location, 'prediction_scores.csv'))
    return df

def get_main_scores(df):
    df = df[df['entity']=='all']
    df = df[df['type']=='entity']
    df.drop(df.columns.difference(['precision','recall', 'f1']), 1, inplace=True)
    return df

def macro_average(df, save_location=''):
    x = df.drop(df.columns.difference(['precision','recall', 'f1']), 1, inplace=False)
    mean = x.mean()
    std = x.std()
    if save_location:
        save_name = list(set(x.index.values))[0]
        mean.to_csv(join(save_location, save_name + "_mean"))
        std.to_csv(join(save_location, save_name + "_std"))
    return mean, std