from abc import ABC, abstractmethod
from image import Image

class ImageFilter(ABC):
    
    @abstractmethod
    def get_image(self) -> Image:
        pass