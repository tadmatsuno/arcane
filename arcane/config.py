"""Read optional external-tool settings without creating configuration on import."""

import json
import os
import shutil


def read_config(filename):
    try:
        with open(filename, encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        return {}


def set_path(filename, key, path):
    path = os.fsdecode(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    settings = read_config(filename)
    settings[key] = path
    if os.path.exists(filename):
        shutil.copy(filename, filename + "_old")
    with open(filename, "w", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2)
        stream.write("\n")
