import torch
import dlib
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from nets import mobilenet_v1
from utils.utils import get_roi, crop_img, get_final_roi, predict_68pts
from utils.transformation import ToTensorUint, NormalizeUint


class Detector(object):
    def __init__(self, cnn=False, with_landmarks=True):
        self.cnn = cnn
        self.with_landmarks = with_landmarks
        if self.cnn:
            self.face_detector = dlib.cnn_face_detection_model_v1('model_params/mmod_human_face_detector.dat')
        else:
            self.face_detector = dlib.get_frontal_face_detector()
        if self.with_landmarks:
            self.transform = transforms.Compose([ToTensorUint(), NormalizeUint(mean=127.5, std=128)])
            checkpoint = torch.load('model_params/phase1_wpdc_vdc.pth.tar',
                                    map_location=lambda storage, loc: storage)['state_dict']
            model = getattr(mobilenet_v1, 'mobilenet_1')(num_classes=62)
            model_dict = model.state_dict()
            for k in checkpoint.keys():
                model_dict[k.replace('module.', '')] = checkpoint[k]
            model.load_state_dict(model_dict)
            if torch.cuda.is_available():
                cudnn.benchmark = True
                model = model.cuda()
            model.eval()
            self.model = model

    def get_eye_area_roi(self, img_path, pixel_threshold=10):
        img_ori = cv2.imread(img_path)
        rgb_image = img_ori[:, :, ::-1]

        rects = self.face_detector(rgb_image, 0)
        if len(rects) == 0:
            return False

        rect = next(iter(rects))
        if self.cnn:
            rect = rect.rect
        roi_box = get_roi(rect, img_ori.shape)
        img = crop_img(img_ori, roi_box)
        if (img.shape[0] < pixel_threshold) or (img.shape[1] < pixel_threshold):
            return False

        if self.with_landmarks:
            img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
            input = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                if torch.cuda.is_available():
                    input = input.cuda()
                param = self.model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = predict_68pts(param, roi_box)
            roi_box = get_final_roi(pts68, img_ori.shape)
            cropped_img = crop_img(img_ori, roi_box)
            cropped_img = cv2.resize(cropped_img, dsize=(64, 32), interpolation=cv2.INTER_LINEAR)
            return cropped_img