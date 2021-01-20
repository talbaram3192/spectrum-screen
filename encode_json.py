import json
import requests
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt


with open('rand_forest.pickle', 'rb') as m:
    MODEL = pickle.load(m)

COLUMNS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons',
       'Sex', 'Jaundice', 'Family_mem_with_ASD']


def get_explanation(pred):  # Dictionary object
    pred2 = np.array(pd.Series(pred)).reshape(1, len(pred))

    explainer = shap.TreeExplainer(MODEL)
    shap_values = explainer.shap_values(pred2)
    shap.initjs()
    shap.summary_plot(shap_values, pred2, feature_names=COLUMNS, show=False)
    plt.savefig('summary.png')

    # shap.force_plot(explainer.expected_value[0],
    #                 shap_values[0][0, :], pred2, feature_names=COLUMNS, show=False)
    # plt.savefig('force_plot.png')


def main():

       my_json = {"A1": 0,"A2": 1,"A3": 4,"A4": 3,"A5": 1, "A6":4, "A7": 2,
               "A8": 0,"A9": 1,"A10": 0,"age_month": 20,"sex": 2,"jaundice": 0,"family_mem_with_ASD": 1}

       # url = 'https://spectrum-screen-inference.herokuapp.com/api/predict'
       url = 'http://127.0.0.1:5000/api/predict'  # for local testing

       pred = requests.post(url, json=json.dumps(my_json))
       print(pred.json())

       # get_explanation(my_json)






if __name__ == '__main__':
    main()
