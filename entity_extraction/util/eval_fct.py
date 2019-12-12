import pandas as pd
import sklearn.metrics as m

from . import evaluation

def evaluate_flat(truth, pred):
    """
    Evaluate Recognition based on two sequences. 
    
    Arguments:
        truth {[str]} -- [sequence of goldstandard labels]
        pred {[str]} -- [sequence of predicted labels]
    
    Returns:
        [pd dataframe] -- [summary of different metrics]
    """
    return evaluation.evaluate_df(truth,pred)

def evaluate_bio(gold_bio, prediction_bio):
    """
    Compare two BIO files, one as goldstandard, second as prediction.
    Wrapper around flat evaluation.
    
    Arguments:
        gold_bio {[str]} -- [goldstandard in BIO format]
        prediction_bio {[str]} -- [prediction in BIO format]
    
    Returns:
        [pd dataframe] -- [summary of different metrics]
    """
    truth, truth_tokens = evaluation.read_bio_file(gold_bio)
    pred, pred_tokens = evaluation.read_bio_file(prediction_bio)
    return evaluate_flat(truth, pred)

def evaluate(actual, prediction):
    """
    Compare lists of lists, one as goldstandard, second as prediction.
    Wrapper around flat evaluation.
    
    Arguments:
        gold_bio {[str]} -- [goldstandard as list of lists]
        prediction_bio {[str]} -- [prediction as list of lists]
    
    Returns:
        [pd dataframe] -- [summary of different metrics]
    """
    truth = [item for sublist in actual for item in sublist]
    pred = [item for sublist in prediction for item in sublist]
    
    return evaluate_flat(truth, pred)

if __name__ == "__main__":
    # Perform a small test to see if everything goes as it should. 
    y_true = 'O O O O B-software I-software O O B-software O O B-software I-software I-software I-software O O O B-software O'.split()
    y_pred = 'O B-software I-software O B-software O O O O O O B-software I-software I-software O O I-software O O B-software'.split()
    print(len(y_true))
    print(len(y_pred))
    df = evaluate_flat(y_true, y_pred)
    print(df)