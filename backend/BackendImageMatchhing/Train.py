from Similarity import Similarity
from FeatureImage import feature_all
import numpy as np
import os
import joblib
import warnings

def save_model(filename, clf):
    with open(filename, 'wb') as f:
        joblib.dump(clf, f, compress=3)

def load_model(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return joblib.load(path)

def load_all(model_path):
    matrix = load_model(model_path + 'matrix.obj')
    list_path = load_model(model_path + 'path.obj')
    return matrix, list_path

class Train:
    def __init__(self, data_path, model_path):
        self.data_path = os.path.abspath(data_path)
        self.model_path = os.path.abspath(model_path)
        self.list_path = self.load_data()

    def load_data(self):
        tmp = []
        for r, d, f in os.walk(self.data_path):
            for file in f:
                if 'jpg' in file or 'png' in file:
                    print(os.path.join(os.path.abspath(self.data_path), file))
                    tmp.append(os.path.join(r, file))
        return tmp

    def run(self):
        list_obj = []
        list_path = []
        for i, path in enumerate(self.list_path):
            try:
                temp = feature_all(path)
                list_obj.append(temp)
                list_path.append(path)
            except:
                # print(path)
                pass
        list_obj = np.array(list_obj)
        save_model(self.model_path + '/matrix.obj', list_obj)
        save_model(self.model_path + '/path.obj', list_path)
        # a = load_model(self.model_path + 'matrix.obj')
        # b = load_model(self.model_path + 'path.obj')
        # print(a.shape)
        # print(b)


if __name__ == '__main__':
    t = Train('Data/', 'Model/')
    t.run()
