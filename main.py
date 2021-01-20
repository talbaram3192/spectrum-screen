from flask import Flask
from flask import jsonify, make_response

app = Flask(__name__)


@app.route("/api/predict", methods=["POST"])
def predict():
    return make_response(jsonify({"prediction": 0, "prediction_probability": 0.98}), 200)


if __name__ == '__main__':
    app.run(debug=True)