import json
import os

config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path) as f:
    config = json.load(f)

def get_Parameter(nametuple, defaultValue=None):
    if isinstance(nametuple, tuple):
        params = config[nametuple[0]]
        for i in range(1, len(nametuple)):
            result = params[nametuple[i]]
        return result
    elif isinstance(nametuple, str):
        return config[nametuple]