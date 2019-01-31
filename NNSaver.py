import json

class NNSaver:

    @staticmethod
    def save(model, file_name):
        json_str = model.to_json()
        with open(file_name, 'w') as outfile:
                json.dump(json_str, outfile)

