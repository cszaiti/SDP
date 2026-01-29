# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np
import os


def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            answers[js["idx"]] = js["target"]
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx, label = line.split()
            predictions[int(idx)] = int(label)
    return predictions


def calculate_scores(answers, predictions):
    Acc = []
    TP, TN, FN, FP = 0, 0, 0, 0
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key] == predictions[key])
        if answers[key] == predictions[key] == 1:
            TP += 1
        elif answers[key] == predictions[key] == 0:
            TN += 1
        elif (answers[key] == 1) and (predictions[key] == 0):
            FN += 1
        else:
            FP += 1

    scores = {}
    scores["Acc"] = np.mean(Acc)
    scores["Recall"] = TP / (TP + FN)
    if (TP + FP) == 0:
        scores["Precision"] = 0
        scores["F1"]=0
    else :
        scores["Precision"] = TP / (TP + FP)
        scores["F1"] = 2 * (
            (scores["Precision"] * scores["Recall"])
            / (scores["Precision"] + scores["Recall"])
        )
    if TN==0:
        scores["Re_N"]=0
        scores["Pre_N"]=0 
    else:    
        scores["Re_N"]=TN/(TN+FP)
        scores["Pre_N"]=TN/(TN+FN) 
    return scores



# def main():
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Evaluate leaderboard predictions for Defect Detection dataset."
#     )
#     parser.add_argument(
#         "--answers", "-a", help="filename of the labels, in txt format."
#     )
#     parser.add_argument(
#         "--predictions",
#         "-p",
#         help="filename of the leaderboard predictions, in txt format.",
#     )

#     args = parser.parse_args()
#     answers = read_answers(args.answers)
#     predictions = read_predictions(args.predictions)
#     scores = calculate_scores(answers, predictions)
#     print(scores)


# if __name__ == "__main__":
#     main()

def evaluators(args):
    import argparse
    checkpoint_prefix = "predictions.txt"
    predictions_dir=os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
    answers_dir=args.test_data_file
    
    answers=read_answers(answers_dir)
    predictions = read_predictions(predictions_dir)    
    scores = calculate_scores(answers, predictions)    
    # print(scores)
    
    return scores