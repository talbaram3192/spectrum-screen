from flask import Flask, request, jsonify, make_response, send_file
import json
import random
import pickle
import config as CFG
import sklearn
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
# import pandas as pd


def load_model():
    return pickle.load(open(CFG.MODEL_FILE, 'rb'))


def load_age_mons_preprocessing():
    return pickle.load(open(CFG.AGE_MONS_PREPROCESSING_FILE, 'rb'))


def load_training_set():
    return pickle.load(open(CFG.TRAINING, 'rb'))


app = Flask(__name__)
model = load_model()
age_mons_preprocessing = load_age_mons_preprocessing()
training = load_training_set()


def convert_json(my_dict):
    new_json = {}
    for num, i in enumerate(my_dict):
        if num <= 8:
            if my_dict[i] == 2 or my_dict[i] == 3 or my_dict[i] == 4:
                new_json[i] = 1
            else:
                new_json[i] = 0
        elif num == 9:
            if my_dict[i] == 0 or my_dict[i] == 1 or my_dict[i] == 2:
                new_json[i] = 1
            else:
                new_json[i] = 0
        elif num == 11:
            if my_dict[i] == 2:
                new_json[i] = random.randint(0, 1)
            else:
                new_json[i] = my_dict[i]
        else:
            new_json[i] = my_dict[i]

    return json.dumps(new_json)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        my_json = request.get_json()
        encoded_dict = convert_json(my_json)
        dictionary = eval(encoded_dict)

        normalize_age_mons = age_mons_preprocessing.transform([[dictionary['age_month']]])[0, 0]

        dictionary['age_month'] = normalize_age_mons
        my_dict = np.array([list(dictionary.values())])

        prediction_probability = model.predict_proba(my_dict)[0]

        if prediction_probability[1] > 0.5:
            prediction_probability = prediction_probability[1]
            predict_spectrum = 1
        else:
            prediction_probability = prediction_probability[0]
            predict_spectrum = 0

        response = {"prediction": predict_spectrum, "prediction_probability": prediction_probability}

        return make_response(jsonify(response), 200)

    except ValueError:
        return 'Bad Request', 400


# @app.route("/api/explain", methods=["POST"])
# def explain():
#     try:
#         my_json = request.get_json()
#         encoded_dict = convert_json(my_json)
#         dictionary = eval(encoded_dict)
#
#         normalize_age_mons = age_mons_preprocessing.transform([[dictionary['age_month']]])[0, 0]
#
#         dictionary['age_month'] = normalize_age_mons
#         pred = np.array([x[1] for x in dictionary.items()])
#
#         exp = LimeTabularExplainer(training.values, feature_names=training.columns, discretize_continuous=True)
#
#         fig = exp.explain_instance(pred, model.predict_proba).as_pyplot_figure()
#         fig.figsize = (30, 10)
#         plt.tight_layout()
#         plt.savefig('explain.png')
#
#         return send_file('explain.png', mimetype='image/png', as_attachment=True)
#
#     except ValueError:
#         return 'Bad Request', 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
