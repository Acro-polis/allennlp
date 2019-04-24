import json
import pickle
import gc


def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)


def load_pickle(path):
    # this makes pickling faster
    gc.disable()
    with open(path, 'rb') as file:
        kg_data = pickle.load(file)
    gc.enable()
    return kg_data

