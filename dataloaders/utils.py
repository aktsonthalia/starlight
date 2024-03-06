import random

class RandomApplyOne:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list
        print(f"going to randomly choose between one of {self.transforms_list}")

    def __call__(self, img):
        transform = random.choice(self.transforms_list)
        return transform(img)
    
    def __repr__(self):
        return self.__class__.__name__ + f"({self.transforms_list})"
    
class IdentityTransform:

    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        return sample
    
    def __repr__(self):
        return self.__class__.__name__ + "()"