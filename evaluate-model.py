#!/usr/bin/env python3

import json
import os
import sys

import cv2
import pandas as pd
from tqdm import tqdm
import yaml

from cnn.network import Network
from cnn.architectures import Architectures


def load_data(datapath):
    train = load_csv(datapath, "train.csv")
    test = load_csv(datapath, "test.csv")
    return train, test


def load_csv(datapath, csvfile):
    csvpath = os.path.join(datapath, csvfile)
    df = pd.read_csv(csvpath)
    return [(os.path.join(datapath, x), y) for x, y in zip(df["image"], df["label"])]


def test_set(model, data, set_name, results=None, limit=None):
    correct, wrong = 0, 0

    if not results:
        results = {}
        for k in [
            "file",
            "dataset",
            "label",
            "inferred",
            "label_confidence",
            "inferred_confidence",
        ]:
            results[k] = []

    if limit and len(data) > limit > 0:
        data = data[:limit]

    status = tqdm(total=len(data))
    for file, label in data:
        img = cv2.imread(file)
        if img is None:
            print("error reading file:", file)
            continue
        data = model.convert(img)
        output = model.predict_verbose_data(data)
        out_label = output[0]
        if out_label == label:
            correct = correct + 1
        else:
            wrong = wrong + 1
        status.update(1)
        status.set_postfix_str(correct / (correct + wrong))

        results["file"].append(file)
        results["dataset"].append(set_name)
        results["label"].append(label)
        results["inferred"].append(out_label)
        outs = output[2].tolist()[0]
        results["label_confidence"].append(outs[model.lmap[label]])
        results["inferred_confidence"].append(outs[model.lmap[out_label]])
    status.close()

    return results


def evaluate_model(params, model_name, limit=None):
    mparam = params[model_name]

    print("loading:")
    print("- architecture:", mparam["arch"])
    print("- weights:", mparam["training"]["outpath"])
    model = Network(Architectures[mparam["arch"]], mparam["training"]["outpath"])

    datapath = mparam["data"]["genpath"]
    print("- data:", datapath)
    train, test = load_data(datapath)

    print("Evaluating test set:")
    results = test_set(model, test, "test", limit=limit)
    print("Evaluating train set:")
    results = test_set(model, train, "train", results=results, limit=limit)

    return pd.DataFrame.from_dict(results)


def generate_summary(results):
    output = {}
    valid_results = results[results["label"] == results["inferred"]]
    error_results = results[results["label"] != results["inferred"]]

    train_valid_n = len(valid_results[valid_results["dataset"] == "train"])
    train_error_n = len(error_results[error_results["dataset"] == "train"])
    output["train_accuracy"] = train_valid_n / (train_valid_n + train_error_n)

    test_valid_n = len(valid_results[valid_results["dataset"] == "test"])
    test_error_n = len(error_results[error_results["dataset"] == "test"])
    output["test_accuracy"] = test_valid_n / (test_valid_n + test_error_n)

    output["overall_accuracy"] = len(valid_results) / (
        len(valid_results) + len(error_results)
    )

    return output


def make_file_path(filename):
    filepath = os.path.dirname(filename)
    if filepath != "":
        os.makedirs(filepath, exist_ok=True)


def run(model_name, results_file, summary_file):
    params = yaml.safe_load(open("params.yaml"))
    results = evaluate_model(params, model_name)

    print("writing", results_file)
    make_file_path(results_file)
    results.sort_values("file").to_csv(results_file, index=False)

    print("writing", summary_file)
    make_file_path(results_file)
    summary = generate_summary(results)
    with open(summary_file, "w") as json_file:
        json.dump(summary, json_file)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(
            "\tpython evaluate-model.py model-name results.csv summary.json\n"
        )
        sys.exit(1)

    run(sys.argv[1], sys.argv[2], sys.argv[3])
