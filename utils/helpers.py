from .constants import SCALE_WIDTH, SCALE_HEIGHT

# Viewport settings
WIDTH = None
HEIGHT = None

def set_width_height(W, H):
    global WIDTH
    global HEIGHT
    WIDTH = W
    HEIGHT = H


def print_info(text):
    print(f"[INFO]: {text}")

def is_within_scale(box):
    (start_x, start_y, end_x, end_y) = box
    if abs(end_x - start_x) < SCALE_WIDTH or abs(end_y - start_y) < SCALE_HEIGHT:
        return False
    return True

def is_within_bounds(box):
    (start_x, start_y, end_x, end_y) = box

    if start_x < 0 or start_y < 0 or end_x > WIDTH or end_y > HEIGHT:
        return False
    return True

def crop_image_from_bounding_box(image, box):
    '''
    To crop an image from its bounding box
    '''
    (start_x, start_y, end_x, end_y) = box
    if start_x < 0 : start_x = 0
    if end_x > WIDTH: end_x = WIDTH
    if start_y < 0: start_y = 0
    if end_y > HEIGHT: end_y = HEIGHT
    return image[start_y:end_y, start_x:end_x]