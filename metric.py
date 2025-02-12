import re
import numpy as np
import collections
import difflib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[\W_]+", " ", text)
    return text

def exact_match_score(prediction, ground_truth): # accuracy
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def character_level_similarity(prediction, ground_truth):
    return difflib.SequenceMatcher(None, prediction, ground_truth).ratio()

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens) # on definit TP est le nombre des terme communs
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def bleu_score(prediction, ground_truth):
    pred_tokens = prediction.split()
    reference_tokens = [ground_truth.split()]
    smoothie = SmoothingFunction().method1
    score = sentence_bleu(reference_tokens, pred_tokens, smoothing_function=smoothie)
    return score

def meteor_metric(prediction, ground_truth):
    return meteor_score([ground_truth.split(' ')], prediction.split(' '))

def compute_metrics(predictions, labels):
    exact_matches = []
    character_diff = []
    f1_scores = []
    bleu_scores = []
    meteor_scores = []

    for pred, gt in zip(predictions, labels):
        exact_matches.append(int(normalize_text(pred) == normalize_text(gt)))
        character_diff.append(character_level_similarity(pred, gt))
        f1_scores.append(f1_score(pred, gt))
        bleu_scores.append(bleu_score(pred, gt))
        meteor_scores.append(meteor_metric(pred, gt))

    print(f'exact_matches: {exact_matches}')
    print(f'character_diff: {character_diff}')
    print(f'f1_scores: {f1_scores}')
    print(f'bleu_scores: {bleu_scores}')
    print(f'meteor_scores: {meteor_scores}')

    return {
        'exact_match': np.mean(exact_matches),
        'character_diff': np.mean(character_diff),
        'f1': np.mean(f1_scores),
        'bleu': np.mean(bleu_scores),
        'meteor': np.mean(meteor_scores),
    }
