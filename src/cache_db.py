from __future__ import annotations
from image import Image
import os
import json

class CacheDb:
    values_dict_filename = "cache_db_values.json"
    def __init__(self, folder_path: str):
        self.cache_folder = folder_path
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        self.cache: dict[str, str] = self.__load_json_file()
        self.load()
        self.individual_save = True

    def __load_json_file(self) -> dict[str, str]:
        path = os.path.join(self.cache_folder, CacheDb.values_dict_filename)
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        return data
    
    def update(self):
        path = os.path.join(self.cache_folder, CacheDb.values_dict_filename)
        with open(path, "w") as f:
            json.dump(self.cache, f)

    def load(self) -> CacheDb:
        files = os.listdir(self.cache_folder)
        paths = [f[:-4] for f in files if f.endswith(".png")]
        for p in paths:
            if not p in self.cache:
                self.cache[p] = ""
        return self
    
    # saving to cache folder, replacing older values or storing info
    # if the old value has info, it is kept
    def store_image(self, cell: Image):
        hash_value = cell.hash()
        if hash_value in self.cache:
            return
        cell.save_to_folder(self.cache_folder)
        if cell.info is not None:
            self.cache[hash_value] = cell.info
            if self.individual_save:
                self.update()