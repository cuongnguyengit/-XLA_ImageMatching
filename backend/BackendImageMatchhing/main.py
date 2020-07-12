# -*- encoding: utf-8 -*-
"""
License: Commercial
Copyright (c) 2019 - present AppSeed.us
"""
from flask import request, jsonify, make_response
import json
import numpy as np
from flask import Flask
from Train import load_model
from FeatureImage import feature_all
from Similarity import Similarity
from flask_cors import CORS

matrix = load_model('Model/matrix.obj')
matrix = matrix.reshape(matrix.shape[0], matrix.shape[2])
list_path = load_model('Model/path.obj')
app = Flask(__name__)
CORS(app)

def get_top(list_distance, length=10):
    index_out = np.argsort(np.array(list_distance))[:length]
    output = []
    for j in index_out:
        output.append(list_path[j])
    return output

def get_list_url_similar(abs_url, length=10):
    fea = feature_all(abs_url)

    list_cosin = Similarity.distance_list(fea, matrix, cosine=True)
    list_eu = Similarity.distance_list(fea, matrix, cosine=False)

    list_url_cosin = get_top(list_cosin.flatten(), length)
    list_url_eu = get_top(list_eu.flatten(), length)

    return list_url_cosin, list_url_eu



@app.route('/', methods=['POST'])
def get_image_urls():
    result = {"list_url_cosine": [], "list_url_eu": []}
    try:
        js_data = request.get_data().decode(encoding='utf-8')
        data = json.loads(js_data)
        print(data)
        abs_url = data["url"]
        length = data.get("len", 10)
        list_url_cosin, list_url_eu = get_list_url_similar(abs_url, length)
        result["list_url_cosine"] = list_url_cosin
        result["list_url_eu"] = list_url_eu
    except Exception as e:
        print(e)
        result = {"error": e}

    res = make_response(jsonify(result))
    res.headers['Access-Control-Allow-Origin'] = '*'
    return res


if __name__ == "__main__":
    # for i in col.find({}):
    #     print(i)
    app.debug = True
    app.run(host="127.0.0.1", port=5000)
