import cv2
from skimage.metrics import structural_similarity as ssim

from utils.helpers import print_info, crop_image_from_bounding_box

from feature_extractor import get_cosine_similarity

ORB = cv2.ORB_create(nfeatures=1000)
BRUTE_FORCE = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def match_people(input_image, box, trackables):
    '''
    Matches people based on their image
    '''
    input_image = crop_image_from_bounding_box(input_image, box)

    if len(trackables) == 0:
        # For first person
        return False, len(trackables), input_image
    
    high_score = float('-inf')
    detected_index = None
    for i, trkble in trackables.items():
        old_image = trkble.image
        score = _get_score(input_image, old_image)
        if score > high_score:
            high_score = score
            detected_index = i

    # cv2.imshow('Matching Image', matching_image)
    print_info(f'High Score for Matching: {high_score}')
    if high_score > 0.8:
        # Person is detected, updating last frame
        cv2.imshow(f'Existing t{detected_index}.jpeg', trackables[detected_index].image)
        cv2.imshow(f'Existing t+1{detected_index}.jpeg', input_image)
        return True, detected_index, input_image
    
    else:
        # Person not found, adding to list
        cv2.imshow(f'New t{detected_index}.jpeg', trackables[detected_index].image)
        cv2.imshow(f'New t+1{detected_index}.jpeg', input_image)
        return False, len(trackables), input_image

def _get_score(input_image, to_match_image):

    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_resize = cv2.resize(input_gray, (250, 250))
    # input_keypoints, input_descriptors = ORB.detectAndCompute(input_resize, None)

    to_match_gray = cv2.cvtColor(to_match_image, cv2.COLOR_BGR2GRAY)
    to_match_resize = cv2.resize(to_match_gray, (250, 250))
    # to_match_keypoints, to_match_descriptors = ORB.detectAndCompute(to_match_resize, None)

    # matches = BRUTE_FORCE.match(input_descriptors, to_match_descriptors)


    # avg_key = (len(to_match_keypoints) + len(input_keypoints)) // 2

    # orb_score = len(matches) / avg_key
    # print(f'ORB Score: {orb_score}')

    ssim_score = ssim(input_resize, to_match_resize)
    # print(f'SSIM Score: {ssim_score}')


    input_hist = cv2.calcHist([input_resize],[0],None,[256],[0,256])
    input_hist = cv2.normalize(input_hist, input_hist).flatten()

    match_hist = cv2.calcHist([to_match_resize],[0],None,[256],[0,256])
    match_hist = cv2.normalize(match_hist, match_hist).flatten()

    bhatt_score = cv2.compareHist(input_hist, match_hist, cv2.HISTCMP_BHATTACHARYYA)
    # print(f'Bhattacharyya Histogram Score: {1 - bhatt_score}')

    corr_score = cv2.compareHist(input_hist, match_hist, cv2.HISTCMP_CORREL)
    # print(f'Correlation Histogram Score: {corr_score}')

    cosine_score = get_cosine_similarity(input_image, to_match_image)
    # print(f'Cosine Similarity Score: {cosine_score}')
    

    # matching_image = cv2.drawMatches(input_resize, input_keypoints, to_match_resize, to_match_keypoints, matches[:50], None, flags=2)

    # return 0.2 * orb_score + 0.2 * ssim_score + 0.3 * corr_score + 0.3 * (1 - bhatt_score)
    return 0.333 * corr_score + 0.333 * (1 - bhatt_score) + 0.333 * cosine_score
