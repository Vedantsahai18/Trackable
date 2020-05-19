import cv2
from skimage.metrics import structural_similarity

from helpers import print_info

class Matcher:

    def __init__(self):
        self.ssim = structural_similarity
        self.people = []

    def add_person(self, person_image):
        self.people.append(person_image)
    
    def match_people(self, input_image, box):
        input_image = self.__crop_image_from_bounding_box(input_image, box)

        if len(self.people) < 1:
            # For first person
            self.add_person(input_image)
            return False, None

        high_score = float('-inf')
        detected_index = None
        for i, p in enumerate(self.people):
            resized = cv2.resize(input_image, (p.shape[1], p.shape[0]))
            score = self.ssim(p, resized, multichannel=True)
            if score > high_score:
                high_score = score
                detected_index = i
            print_info(f'Index: {i} | Score: {score}')

        if high_score > 0.15:
            # Person is detected, updating last frame
            self.people[detected_index] = input_image
            return True, detected_index
        else:
            # Person not found, adding to list
            self.people.append(input_image)
            return False, None

    def __crop_image_from_bounding_box(self, image, box):
        (start_x, start_y, end_x, end_y) = box
        return image[start_y:end_y, start_x:end_x]
