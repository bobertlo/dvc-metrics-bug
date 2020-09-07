#!/usr/bin/env python3

import os
import shutil
import sys
import yaml

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cnn.architectures import Architectures
from cnn.datautil import Data


def load_classes_flat(basedir, exclude=None, classes=None):

    data = Data(basedir)

    if not classes:
        classes = data.classes()

    if exclude:
        for e in exclude:
            if e in classes:
                classes.remove(e)

    screen_files = {}
    for c in classes:
        screen_files[c] = data.subclassfiles(c)

    total = 0
    for k in screen_files:
        n = len(screen_files[k])
        if n < 2:
            print("warning", k, "length < 2")
        total = total + n

    print(total, "files found")
    return screen_files


def split_screen_files(screen_files, class_limit=None):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for k in screen_files:
        subfiles = screen_files[k]
        subfiles.sort()
        if len(subfiles) < 2:
            # add to both train and test because it will break the labels otherwise!
            X_train.append(subfiles[0])
            y_train.append(k)
            X_test.append(subfiles[0])
            y_test.append(k)
            continue
        sublabels = []
        for _ in range(len(subfiles)):
            sublabels.append(k)
        if class_limit and len(subfiles) > class_limit:
            subfiles, _, sublabels, _ = train_test_split(
                subfiles, sublabels, train_size=class_limit, random_state=42
            )
        subX_train, subX_test, suby_train, suby_test = train_test_split(
            subfiles, sublabels, test_size=0.2, random_state=42
        )
        X_train.extend(subX_train)
        X_test.extend(subX_test)
        y_train.extend(suby_train)
        y_test.extend(suby_test)

    return (X_train, X_test, y_train, y_test)


def filter_copy(infile, outfile):
    if not os.path.exists(outfile):
        shutil.copy(infile, outfile)


def filter_crop(infile, outfile, crop_function):
    img = cv2.imread(infile)
    img = crop_function(img)
    cv2.imwrite(outfile, img)


write_filters = {
    "copy": filter_copy,
}


def dump_table_f(basedir, csv_prefix, X, y, f=filter_copy):
    X_rel = []
    files = []
    files_written = 0
    data_prefix = "images"

    for i in tqdm(range(len(X))):
        # prepare path for file
        thispath = os.path.join(basedir, data_prefix, y[i])
        os.makedirs(thispath, exist_ok=True)
        thisfile = os.path.join(thispath, os.path.basename(X[i]))

        # add file and label to metadata
        files.append(thisfile)
        X_rel.append(os.path.join(data_prefix, y[i], os.path.basename(X[i])))

        # write file (if it doesn't exist already)
        if os.path.exists(thisfile):
            continue
        f(X[i], thisfile)
        files_written = files_written + 1

    # write metadata to csv
    df = pd.DataFrame(list(zip(X_rel, y)), columns=["image", "label"])
    df.to_csv(os.path.join(basedir, csv_prefix + ".csv"), index=False)

    return files, files_written


def clean_fileset(basepath, files, subpath=None):
    removed = []
    if subpath:
        thispath = os.path.join(basepath, subpath)
    else:
        subpath = ""
        thispath = basepath
    for f in os.listdir(thispath):
        thisfile = os.path.join(thispath, f)
        if os.path.isdir(thisfile):
            removed.extend(clean_fileset(basepath, files, os.path.join(subpath, f)))
        elif os.path.isfile(thisfile) and not thisfile in files:
            removed.append(thisfile)
            os.remove(thisfile)
    return removed


def dump_sets(outpath, dataset, f=filter_copy):
    X_train, X_test, y_train, y_test = dataset
    files = []
    written_total = 0
    files_written, n_written = dump_table_f(outpath, "train", X_train, y_train, f=f)
    files.extend(files_written)
    written_total = written_total + n_written

    files_written, n_written = dump_table_f(outpath, "test", X_test, y_test, f=f)
    files.extend(files_written)
    written_total = written_total + n_written
    print(written_total, "files written")

    files.append(os.path.join(outpath, "train.csv"))
    files.append(os.path.join(outpath, "test.csv"))

    removed = clean_fileset(outpath, set(files))
    print("fileset clean,", len(removed), "extra files removed")
    return files


def training_gen(params):
    data = params["data"]
    inpath = data["rawpath"]
    outpath = data["genpath"]
    exclude = data.get("exclude", None)
    classes = data.get("classes", None)
    write_f = data.get("write_filter", "copy")
    seed = data.get("seed", 20200827)
    class_limit = data.get("class_limit", None)

    if write_f == "crop":
        crop_function = Architectures[params["arch"]].crop
        write_func = lambda infile, outfile: filter_crop(infile, outfile, crop_function)
    else:
        write_func = write_filters[write_f]

    screen_files = load_classes_flat(inpath, exclude=exclude, classes=classes)
    np.random.seed(seed=seed)
    dataset = split_screen_files(screen_files, class_limit=class_limit)
    dump_sets(outpath, dataset, f=write_func)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py model-repro.yml\n")
        sys.exit(1)

    mparams = yaml.safe_load(open("params.yaml")).get(sys.argv[1])
    if not mparams:
        sys.stderr.write("Invalid repro.yml. aborting")
        sys.exit(1)

    training_gen(mparams)
