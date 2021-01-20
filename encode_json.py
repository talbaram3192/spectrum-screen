import json
import random
import requests

def convert_json(my_json):
    new_json = {}
    json_get = json.loads(my_json)
    for num, i in enumerate(json_get):
        if num <= 8:
            if json_get[i] == 2 or json_get[i] == 3 or json_get[i] == 4:
                new_json[i] = 1
            else:
                new_json[i] = 0
        elif num == 9:
            if json_get[i] == 0 or json_get[i] == 1 or json_get[i] == 2:
                new_json[i] = 1
            else:
                new_json[i] = 0
        elif num == 11:
            if json_get[i] == 2:
                new_json[i] = random.randint(0, 1)
            else:
                new_json[i] = json_get[i]
        else:
            new_json[i] = json_get[i]

    return json.dumps(new_json)


def main():

       my_json = {"A1": 0,"A2": 1,"A3": 4,"A4": 3,"A5": 1, "A6":4, "A7": 2,
               "A8": 0,"A9": 1,"A10": 0,"age_month": 20,"sex": 2,"jaundice": 0,"family_mem_with_ASD": 1}
       new = convert_json(json.dumps(my_json))

       print(new)
       url = 'https://spectrum-screen-inference.herokuapp.com/api/predict'

       pred = requests.post(url, json=json.dumps(new))

       print(pred.json())




if __name__ == '__main__':
    main()
