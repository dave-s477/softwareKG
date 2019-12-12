#!/bin/python3

# Copy partially copied from: http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

from collections import namedtuple
from copy import deepcopy
import argparse
import pandas as pd
import sklearn.metrics

Entity = namedtuple("Entity", "e_type start_offset end_offset")


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges

    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().

    Examples:

    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results['correct']
    incorrect = results['incorrect']
    partial = results['partial']
    missed = results['missed']
    spurious = results['spurious']

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def compute_precision_recall_f1(results, partial_or_type=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results['partial']
    correct = results['correct']

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall
    if precision + recall == 0:
        results["f1"] = 0
    else:
        results["f1"] = 2 * ((precision * recall) / (precision + recall))
    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {key: compute_precision_recall_f1(value, True) for key, value in results.items() if
                 key in ['partial', 'ent_type']}
    results_b = {key: compute_precision_recall_f1(value) for key, value in results.items() if
                 key in ['strict', 'exact']}

    results = {**results_a, **results_b}

    return results


def compute_metrics(true_named_entities, pred_named_entities, target_tags_no_schema):
    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}
    

    # overall results
    evaluation = {'strict': deepcopy(eval_metrics),
                  'ent_type': deepcopy(eval_metrics),
                  'partial': deepcopy(eval_metrics),
                  'exact': deepcopy(eval_metrics)}

    # results by entity type
    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in target_tags_no_schema}

    true_which_overlapped_with_pred = []  # keep track of entities that overlapped

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # Check each of the potential scenarios in turn. See 
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation. 

        # Scenario I: Exact match between true and pred

        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1
            evaluation['exact']['correct'] += 1
            evaluation['partial']['correct'] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['exact']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['partial']['correct'] += 1

        else:

            # check for overlaps with any of the true entities
            pred_range = range(pred.start_offset, pred.end_offset + 1)

            for true in true_named_entities:

                true_range = range(true.start_offset, true.end_offset + 1)

                # Scenario IV: Offsets match, but entity type is wrong

                if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset \
                        and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1
                    evaluation['partial']['correct'] += 1
                    evaluation['exact']['correct'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['partial']['correct'] += 1
                    evaluation_agg_entities_type[true.e_type]['exact']['correct'] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

                # check for an overlap i.e. not exact boundary match, with true entities

                elif find_overlap(true_range, pred_range):

                    true_which_overlapped_with_pred.append(true)

                    # Scenario V: There is an overlap (but offsets do not match
                    # exactly), and the entity type is the same.
                    # 2.1 overlaps with the same entity type

                    if pred.e_type == true.e_type:

                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['correct'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        found_overlap = True
                        break

                    # Scenario VI: Entities overlap, but the entity type is 
                    # different.

                    else:
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        # Results against the true entity

                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        # Results against the predicted entity

                        # evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1

                        found_overlap = True
                        break

            # Scenario II: Entities are spurious (i.e., over-generated).

            if not found_overlap:
                # overall results
                evaluation['strict']['spurious'] += 1
                evaluation['ent_type']['spurious'] += 1
                evaluation['partial']['spurious'] += 1
                evaluation['exact']['spurious'] += 1

                # aggregated by entity type results
                evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['ent_type']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['partial']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['exact']['spurious'] += 1

    # Scenario III: Entity was missed entirely.

    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1
            evaluation['partial']['missed'] += 1
            evaluation['exact']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['partial']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['exact']['missed'] += 1

    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.

    for eval_type in evaluation:
        evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

    # Compute 'possible', 'actual', and precision and recall on entity level 
    # results. Start by cycling through the accumulated results.

    for entity_type, entity_level in evaluation_agg_entities_type.items():

        # Cycle through the evaluation types for each dict containing entity
        # level results.

        for eval_type in entity_level:

            evaluation_agg_entities_type[entity_type][eval_type] = compute_actual_possible(
                entity_level[eval_type]
            )

    return evaluation, evaluation_agg_entities_type

def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of labels
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O' or token_tag == '<PAD>':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens)-1))

    return named_entities



def read_bio_file(filename):
    tags = []
    tokens = []
    with open(filename, 'r') as ff:
        for line in ff:
            line = line.rstrip()
            if line.startswith("-DOCSTART-"):
                ls = line.split(":")
                #if len(ls) != 2:
                #    raise Exception("Missing document name for new document in BIO format")
            else:
                ls = line.split(' ')
                # allow confidence values to be present in the file
                if len(ls) >= 2:
                    tags.append(ls[1].strip())
                    tokens.append(ls[0])
    return tags, tokens

def results_to_dataframe(res, entity='all'):
    r = pd.DataFrame(columns = ['correct', 'incorrect', 'partial', 'missed', 'spurious', 'actual', 'possible', 'precision', 'recall', 'f1'])

    idx = 0
    for rtype in res.keys():
        r.loc[rtype] = list(res[rtype].values())
    r.insert(0, 'entity', entity)
    return r

def tag_based_evaluation(truth, pred, entities):
    r2 = pd.DataFrame(columns = ['entity', 'type', 'correct', 'incorrect', 'partial', 'missed', 'spurious', 'actual', 'possible', 'precision', 'recall', 'f1'])
    c = sklearn.metrics.classification_report(truth, pred, output_dict=True)
    entity_names = []
    for entity in entities:
        entity_names.append('B-'+entity)
        entity_names.append('I-'+entity)

    for entity in entity_names:
        if entity in c:
            m = c[entity]
            r2.loc['strict_' + entity] = [entity, 'tag', None, None, None, None, None, None, 
                                        m['support'],
                                        m['precision'],
                                        m['recall'],
                                        m['f1-score']]
        else:
            r2.loc['strict_' + entity] = [entity, 'tag', None, None, None, None, None, None, 
                                        0,
                                        0,
                                        0,
                                        0]
            
    r2.loc['strict'] = ['all', 'tag', None, None, None, None, None, None, 
                                    sum(r2.possible),
                                    sum(r2.possible*r2.precision)/sum(r2.possible),
                                    sum(r2.possible*r2.recall)/sum(r2.possible),
                                    sum(r2.possible*r2.f1)/sum(r2.possible)]
    return r2

def evaluate_df(truth, pred):
    truth_e = collect_named_entities(truth)
    pred_e = collect_named_entities(pred)

    entities = ['software']

    results, results_agg = compute_metrics(truth_e, pred_e, entities)
    res = compute_precision_recall_wrapper(results)
    
    
    final_result_list = []
    for entity in entities:
        res2 = compute_precision_recall_wrapper(results_agg[entity])
        r = results_to_dataframe(res2, entity=entity)
        final_result_list.append(r)
    

    r = results_to_dataframe(res)
    final_result_list.append(r)

    df = pd.concat(final_result_list)
    df.insert(1, 'type', 'entity')
    #print(df)
    df2 = tag_based_evaluation(truth, pred, entities)
    #print(df2)
    #print(df.append(df2))
    return df.append(df2, sort=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Runs NER evaluation on bio formatted files")

    parser.add_argument("truthfile", nargs=1 ,help="Filename of the true NER tagged sequence in BIO format")
    parser.add_argument("predictionfile", nargs=1, help="Filename of the predicted NER tagged sequence in BIO format")

  
    args = parser.parse_args()
    # print(args)
    truth, truth_tokens = read_bio_file(args.truthfile[0])
    pred, pred_tokens = read_bio_file(args.predictionfile[0])

    if len(truth) != len(pred):
        print(len(truth))
        print(len(pred))
        for i in range(len(truth_tokens)):
            if truth_tokens[i] != pred_tokens[i]:
                print("not in pred: '%s'" %truth_tokens[i])
                print("found instead: '%s'" %pred_tokens[i])
                print("index: %d" %i)
                break
        raise Exception("Truth and prediction must be of same length")

    
    print(evaluate_df(truth,pred))
