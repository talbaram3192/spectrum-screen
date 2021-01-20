import json
import requests
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import webbrowser
import base64
from IPython.display import display, HTML

with open('ml-model/toddler_autism_model_logistic_regression.pickle', 'rb') as m:
    MODEL = pickle.load(m)

COLUMNS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons',
       'Sex', 'Jaundice', 'Family_mem_with_ASD']

TRAINING = pd.read_csv('training.csv')


def get_explanation(pred):  # Dictionary object
    pred2 = json.loads(pred)
    pred3 = np.array([x[1] for x in pred2.items()])

    exp = LimeTabularExplainer(TRAINING.values, feature_names=COLUMNS, discretize_continuous=True)

    fig = exp.explain_instance(pred3, MODEL.predict_proba).as_pyplot_figure()
    fig.figsize = (30, 10)
    plt.tight_layout()
    plt.savefig('force_plot.png')


def main():

       my_json = {"A1": 0,"A2": 1,"A3": 4,"A4": 3,"A5": 1, "A6":4, "A7": 2,
               "A8": 0,"A9": 1,"A10": 0,"age_month": 20,"sex": 2,"jaundice": 0,"family_mem_with_ASD": 1}

       # url = 'https://spectrum-screen-inference.herokuapp.com/api/predict'
       url = 'http://127.0.0.1:5000/api/explain'  # for local testing

       # pred = requests.post(url, json=json.dumps(my_json))
       j = json.dumps(my_json)
       png = requests.post(url, json=j)

       print(type(png.content))
       with open('new.png', 'wb') as f:
           f.write(png.content)

       # print(pred.json())

       # get_explanation(json.dumps(my_json))






if __name__ == '__main__':
    main()
