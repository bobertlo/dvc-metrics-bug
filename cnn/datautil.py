import hashlib
import os
import pickle
import re

import cv2
from tqdm import tqdm


class Data:
    def __load_jpegs(self, dirname):
        flist = []
        for f in os.listdir(dirname):
            fpath = os.path.join(dirname, f)
            if not os.path.isfile(fpath):
                continue
            flist.append(fpath)
        return flist

    def __load_tree(self, basename, pathname):
        thispath = os.path.join(basename, pathname)
        classname = pathname.replace("/", "_")

        # load jpegs in directory into self.fdict[classname]
        jpegs = self.__load_jpegs(thispath)
        if len(jpegs) > 0:
            self.fdict[classname] = jpegs

        # recursively load subdirectories, storing the subclass
        # names in self.subdict[classname]
        subclasses = [classname]
        for name in os.listdir(thispath):
            if os.path.isdir(os.path.join(thispath, name)):
                sub = self.__load_tree(basename, os.path.join(pathname, name))
                subclasses.extend(sub)
        self.subdict[classname] = subclasses

        # used for recursion, ignored elsewhere in the class
        return subclasses

    def __init__(self, directory):
        self.basename = directory
        self.fdict = {}
        self.subdict = {}
        self.__load_tree(directory, "")

    def reload(self):
        self.__load_tree(self.basename, "")

    def classes(self):
        return list(self.fdict.keys())

    def reclasses(self, restring):
        pattern = re.compile(restring)
        classes = set()
        for c in self.subdict.keys():
            if pattern.match(c):
                for subclass in self.subdict[c]:
                    classes.add(subclass)
        return list(classes)

    def classfiles(self, key):
        files = []
        if key in self.fdict:
            files = self.fdict[key]
        return files

    def subclassfiles(self, key):
        subclasses = self.subdict[key]
        files = []
        for c in subclasses:
            if c in self.fdict:
                files.extend(self.fdict[c])
        return files

    def reclassfiles(self, restring):
        files = []
        for c in self.reclasses(restring):
            if c in self.fdict:
                files.extend(self.fdict[c])
        return files

    def subclasses(self, k):
        if k in self.subdict.keys():
            return self.subdict[k]
        else:
            return []

    def files(self):
        return self.subclassfiles("")


class Cache:
    def load(self):
        data = Data(self.basedir)
        all_files = data.subclassfiles("")
        new_cache = {}
        self.keys = set()
        for i in tqdm(range(len(all_files))):
            f = all_files[i]
            m = self.fn(f)
            new_cache[f] = m
            self.keys.add(m)
        self.cache = new_cache

    def update(self):
        data = Data(self.basedir)
        all_files = data.subclassfiles("")
        new_cache = {}
        self.keys = set()
        for i in tqdm(range(len(all_files))):
            f = all_files[i]
            if f in self.cache:
                new_cache[f] = self.cache[f]
                self.keys.add(self.cache[f])
            else:
                m = self.fn(f)
                if m is None:
                    continue
                else:
                    new_cache[f] = m
                    self.keys.add(m)
        self.cache = new_cache
        self.save_pickle()

    def __init__(self, fn, basedir, cachefile=None):
        self.cachefile = cachefile
        self.basedir = basedir
        self.fn = fn
        if cachefile:
            if os.path.exists(cachefile):
                self.load_pickle()
                self.update()
                self.save_pickle()
            else:
                self.load()
                self.save_pickle()
        else:
            self.load()

    def save_pickle(self):
        # Store data (serialize)
        if self.cachefile is not None:
            with open(self.cachefile, 'wb') as handle:
                pickle.dump(self.cache, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self):
        with open(self.cachefile, 'rb') as handle:
            self.cache = pickle.load(handle)

    def lookup(self, file):
        if file in self.cache:
            return self.cache[file]
        else:
            m = self.fn(file)
            self.cache[file] = m
            return m

    def add(self, file, m):
        self.cache[file] = m
        self.keys.add(m)

    def reload(self):
        self.load


def file_hash(f):
    img = cv2.imread(f)
    if img is None:
        return None
    return hashlib.sha256(img.data).digest()


def file_hash_hex(f):
    img = cv2.imread(f)
    if img is None:
        return None
    return hashlib.sha256(img.data).hexdigest()


def HashCache(basedir, cachefile=None):
    return Cache(file_hash, basedir, cachefile=cachefile)


def HexHashCache(basedir, cachefile=None):
    return Cache(file_hash_hex, basedir, cachefile=cachefile)
