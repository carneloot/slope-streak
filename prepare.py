
from os import listdir, path

from util import CROP_SIZE, DATASET_DIRECTORY, IMAGES_DIRECTORY

import cv2 as cv
from pprint import pprint as pp

LEFT_ARROW_CODE = 2424832
RIGHT_ARROW_CODE = 2555904
UP_ARROW_CODE = 2490368

image_paths = []

raw_image_paths = listdir(IMAGES_DIRECTORY)
for raw_image_path in raw_image_paths:
    if raw_image_path.startswith('_'):
        continue

    image_paths.append(f'{IMAGES_DIRECTORY}/{raw_image_path}')

for image_path in image_paths:
    print(f'Opening \'{image_path}\'...')

    image = cv.imread(image_path, 0)

    print('Image opened!')

    height, width = image.shape

    base_filename = path.splitext(path.basename(image_path))[0]

    print(width, height)

    x = 0
    while x + CROP_SIZE < width:

        y = 0
        while y + CROP_SIZE < height:
            cropped_image = image[y:y + CROP_SIZE, x:x + CROP_SIZE]

            window_name = f'{x}_{y}'

            cv.imshow(window_name, cropped_image)
            cv.moveWindow(window_name, 100, 100)

            pressed = cv.waitKeyEx(0)

            cv.destroyWindow(window_name)

            has_slope_streak = None
            if pressed == LEFT_ARROW_CODE:
                has_slope_streak = 0
            elif pressed == RIGHT_ARROW_CODE:
                has_slope_streak = 1
            elif pressed == UP_ARROW_CODE:
                y += CROP_SIZE
                continue

            cv.imwrite(f'{DATASET_DIRECTORY}/{base_filename}-{CROP_SIZE}_{x}_{y}_{has_slope_streak}.jpg', cropped_image)

            y += CROP_SIZE

        x += CROP_SIZE

