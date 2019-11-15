import torch
from nets.tinyVGG import SimpleVGG
from utils.transformation import data_transforms
from argparse import ArgumentParser
from detection import Detector
from glob import glob


classes2name = {
    0 : "with glasses",
    1 : "without glasses"
}

def inference_on_single_image(img, model, transform):
    input = transform(img).unsqueeze(0)
    with torch.no_grad():
        if torch.cuda.is_available():
            input = input.cuda()
        _, predicted = torch.max(model(input).data, 1)
        predicted = predicted.item()
    return predicted

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--img_folder', type=str, default='example_data_glasses/with_glasses',
                        help='The path to the folder which contains images.')
    parser.add_argument('--model_weights', type=str, default='trained_modelparams/best_weights.pth',
                        help='The path to the weights for simple neural network.')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of workers')
    parser.add_argument('--time', action='store_true')
    args = parser.parse_args()

    img_paths = glob(args.img_folder + "/*")

    model = SimpleVGG(2)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()

    detector_default = Detector()
    detector_cnn = Detector(cnn=True)

    if args.time:
        all_time = 0.0
        n = 0
        from time import time

    for img_path in img_paths:
        cropped_img = detector_default.get_eye_area_roi(img_path)
        if cropped_img is False:
            cropped_img = detector_cnn.get_eye_area_roi(img_path)
        if cropped_img is not False:
            if args.time:
                begin = time()
            predict = inference_on_single_image(cropped_img, model, data_transforms['val'])
            if args.time:
                all_time += (time() - begin)
                n += 1
            if predict == 0:
                print(img_path)
    if args.time:
        print("Average time only for inference classification model (without detection) - {}".format(all_time/n))
        # else:
        #     print("Not found face by path - {}".format(img_path))