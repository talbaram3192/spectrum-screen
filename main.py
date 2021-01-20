from flask import Flask, request
from flask import jsonify, make_response
import json
import random

app = Flask(__name__)


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
    my_json = request.get_json()
    my_json = json.loads(my_json)
    print('Before encoding', my_json)
    encoded_dict = convert_json(my_json)
    print('After encoding- ', encoded_dict)

    # return make_response(jsonify({"prediction": 0, "prediction_probability": 0.98}), 200)
    return make_response(jsonify(encoded_dict), 200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)