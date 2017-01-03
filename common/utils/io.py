import json
import os
import pickle

import hickle as hkl

import common.utils.time as time


def load_hkl_file(filename):
    hkl_filename = filename + '.hkl'
    if os.path.isfile(hkl_filename):
        start = time.get_seconds()
        data = hkl.load(hkl_filename)
        print 'Loaded %s in %ds' % (hkl_filename, time.get_seconds() - start)
        return data
    return None


def save_hkl_file(filename, data):
    hkl_filename = filename + '.hkl'
    try:
        hkl.dump(data, hkl_filename, mode="w")
        return True
    except Exception:
        if os.path.isfile(filename):
            os.remove(hkl_filename)


def save_pickle_file(filename, data):
    start = time.get_seconds()
    filename = filename + '.pickle'
    print 'Dumping to %s' % filename,
    with open(filename) as f:
        pickle.dump(data, f)
        print '%ds' % (time.get_seconds() - start)


def load_pickle_file(filename):
    filename = filename + '.pickle'
    if os.path.isfile(filename):
        print 'Loading %s ...' % filename,
        with open(filename) as f:
            start = time.get_seconds()
            data = pickle.load(f)
            print '%ds' % (time.get_seconds() - start)
            return data
    return None


def save_json_file(filename, data):
    start = time.get_seconds()
    filename = filename + '.json'
    print 'Dumping to %s' % filename,
    with open(filename, 'w') as f:
        json.dump(data, f)
        print '%ds' % (time.get_seconds() - start)


def load_json_file(filename):
    filename = filename + '.json'
    if os.path.isfile(filename):
        print 'Loading %s ...' % filename,
        with open(filename, 'r') as f:
            start = time.get_seconds()
            data = json.load(f)
            print '%ds' % (time.get_seconds() - start)
            return data
    return None