from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
from image import Image
from filter_abc import ImageFilter

class Region:
    def __init__(self, center: tuple[float, float]):
        self.center: tuple[float, float] = center
        self.x_min: int = 5000
        self.x_max: int = 0
        self.y_min: int = 5000
        self.y_max: int = 0
        self.points: set[tuple[int, int]] = set()
        self.closest_to_center_point: tuple[int, int] | None = None
        self.closest_to_center_point_distance: float = float('inf')

    def get_h_w(self) -> tuple[int, int]:
        width = self.x_max - self.x_min + 1
        height = self.y_max - self.y_min + 1
        return height, width
    
    def has_point(self, x: int, y: int) -> bool:
        return (x, y) in self.points
    
    def size(self) -> int:
        return len(self.points)

    @staticmethod
    def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def add_point(self, x: int, y: int) -> None:
        self.points.add((x, y))
        if x < self.x_min:
            self.x_min = x
        if x > self.x_max:
            self.x_max = x
        if y < self.y_min:
            self.y_min = y
        if y > self.y_max:
            self.y_max = y

        distance_to_center = Region.distance((x, y), self.center)
        # print(distance_to_center)
        if distance_to_center < self.closest_to_center_point_distance:
            self.closest_to_center_point_distance = distance_to_center
            self.closest_to_center_point = (x, y)

class Decomposer(ImageFilter):
    def __init__(self, image: Image):
        self.regions: list[Region] = []
        self.mapped_points: set[tuple[int, int]] = set()
        self.image: Image = image
        self.find_regions()

    def erase_small_regions(self, percentage: float = 0.1) -> Decomposer:
        h, w = self.image.get_h_w()
        limit_h = h * percentage
        limit_w = w * percentage
        # print("limits", limit_h, limit_w)
        # print([region.get_h_w() for region in self.regions])
        big_regions: list[Region] = []
        for region in self.regions:
            h, w = region.get_h_w()
            if h > limit_h or w > limit_w:
                big_regions.append(region)
        self.regions = big_regions
        return self

    def erase_transverse_regions(self, max_width: int, max_height: int) -> Decomposer:
        def is_transverse(region: Region) -> bool:
            width, height = region.get_h_w()
            return width > max_width or height > max_height
        self.regions = [region for region in self.regions if not is_transverse(region)]
        return self

    def erase_regions_outside_bounds(self, percentage: float = 0.8) -> Decomposer:
        height, width = self.image.get_h_w()
        limit = (min(height, width) / 2) * percentage
        # print([r.closest_to_center_point_distance for r  in self.regions])
        def is_outside_bounds(region: Region) -> bool:
            if region.closest_to_center_point is None:
                return True
            return region.closest_to_center_point_distance > limit
        self.regions = [region for region in self.regions if not is_outside_bounds(region)]
        return self

    def get_image(self) -> Image:
        if not self.regions:
            return Image(self.image).set_data(np.array([], dtype=np.uint8))
        height, width = self.image.get_h_w()

        img = np.zeros((height, width), dtype=np.uint8)
        for region in self.regions:
            for (x, y) in region.points:
                img[y, x] = 255
        return Image(self.image).set_data(img)

    @staticmethod
    def get_neighbors_8(x: int, y: int) -> list[tuple[int, int]]:
        return [
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1),
            (x - 1, y - 1), (x + 1, y - 1),
            (x - 1, y + 1), (x + 1, y + 1)
        ]
    
    @staticmethod
    def get_neighbors_4(x: int, y: int) -> list[tuple[int, int]]:
        return [
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1)
        ]

    def find_regions(self) -> None:
        img = self.image.data
        self.regions = []
        self.mapped_points = set()
        h, w = img.shape
        for y in range(h):
            for x in range(w):
                if img[y, x] == 255 and (x, y) not in self.mapped_points:
                    region = Region(center=(w/2, h/2))
                    self.__explore_region(img, x, y, region)
                    self.regions.append(region)
    
    def __explore_region(self, img: NDArray[np.uint8], x: int, y: int, region: Region) -> None:
        h, w = img.shape
        stack: list[tuple[int, int]] = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in self.mapped_points:
                continue
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                continue
            if img[cy, cx] != 255:
                continue
            region.add_point(cx, cy)
            self.mapped_points.add((cx, cy))
            # Adicionar vizinhos ao stack
            neighbors = Decomposer.get_neighbors_8(cx, cy)
            stack.extend(neighbors)
