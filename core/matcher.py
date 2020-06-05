import cv2

from utils.helpers import print_info, crop_image_from_bounding_box
from .feature_extractor import get_cosine_similarity


def match_people(input_image, box, trackables):
    '''
    Matches people based on their image.
    '''
    
    input_image = crop_image_from_bounding_box(input_image, box)

    if len(trackables) == 0:
        # For first person
        return False, len(trackables), input_image
    
    high_score = float('-inf')
    detected_index = None
    for i, trkble in trackables.items():
        if trkble.is_being_tracked():
            old_image = trkble.image
            score = _get_score(input_image, old_image)
            if score > high_score:
                high_score = score
                detected_index = i

    print_info(f'High Score for Matching: {high_score}')
    if high_score > 0.8:
        # Person is detected, updating last frame
        return True, detected_index, input_image
    else:
        # Person not found, adding to list
        return False, len(trackables), input_image

def _get_score(input_image, to_match_image):
    '''
    Returns the similarity score between the two images.
    '''

    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_resize = cv2.resize(input_gray, (250, 250))

    to_match_gray = cv2.cvtColor(to_match_image, cv2.COLOR_BGR2GRAY)
    to_match_resize = cv2.resize(to_match_gray, (250, 250))

    input_hist = cv2.calcHist([input_resize],[0],None,[256],[0,256])
    input_hist = cv2.normalize(input_hist, input_hist).flatten()

    match_hist = cv2.calcHist([to_match_resize],[0],None,[256],[0,256])
    match_hist = cv2.normalize(match_hist, match_hist).flatten()

    bhatt_score = cv2.compareHist(input_hist, match_hist, cv2.HISTCMP_BHATTACHARYYA)

    corr_score = cv2.compareHist(input_hist, match_hist, cv2.HISTCMP_CORREL)

    cosine_score = get_cosine_similarity(input_image, to_match_image)

    return 0.333 * corr_score + 0.333 * (1 - bhatt_score) + 0.333 * cosine_score
