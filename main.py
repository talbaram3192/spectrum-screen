from flask import Flask
from flask import jsonify, make_response
import pickle
import config as CFG
import sklearn


def load_model():
    return pickle.load(open(CFG.MODEL_FILE, 'rb'))


app = Flask(__name__)
model = load_model()


@app.route("/api/predict", methods=["POST"])
def predict():
    return make_response(jsonify({"prediction": 0, "prediction_probability": 0.98}), 200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
