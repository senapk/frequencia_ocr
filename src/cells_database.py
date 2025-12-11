from __future__ import annotations
from src.image import Image
import os

class CellsDatabase:
    def __init__(self, folder_path: str):
        self.cache_folder = folder_path
        self.cache: dict[str, Image] = {}

    def load_folder(self):
        files = os.listdir()
        paths = [os.path.join(self.cache_folder, f) for f in files if f.endswith(".png")]
        for p in paths:
            cell = Image().load_from_file(p)
            self.cache[cell.hash()] = cell

    # saving to cache folder, replacing older values or storing info
    def save_cell(self, cell: Image):
        hash_value = cell.hash()
        files = os.listdir(self.cache_folder)
        found = [x for x in files if x.startswith(hash_value)]
        if len(found) == 1:
            os.remove(os.path.join(self.cache_folder, found[0]))
        cell.save_to_folder(self.cache_folder)