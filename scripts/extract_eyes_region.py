import os
import cv2
from detection import Detector
from argparse import ArgumentParser
from tqdm import tqdm
from utils.utils import create_folder, obtain_paths


def extract_regions_in_folder(img_names, img_folder, output_folder):
    not_found_face = 0
    detector = Detector()
    for img_name in tqdm(img_names):
        cropped_img = detector.get_eye_area_roi(os.path.join(img_folder, img_name))
        if cropped_img is not False:
            cv2.imwrite(str(output_folder/img_name), cropped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        else:
            not_found_face += 1
    print('The detector could not detect faces in {} images'.format(not_found_face))
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--img_folder', type=str, default='example_data_glasses/with_glasses',
                        help='The path to the folder which contains images with faces.')
    parser.add_argument('--output_folder', type=str, default='./output_images',
                        help='The path to the folder which contains images with faces.')
    args = parser.parse_args()

    output_folder = create_folder(args.output_folder)
    paths = obtain_paths(args.img_folder)

    extract_regions_in_folder(paths, args.img_folder, output_folder)
