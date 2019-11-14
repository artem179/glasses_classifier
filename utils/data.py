import os
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# class EyeGlassesDataset(Dataset):
#     def __init__(self):




def download_images_labels(img_dir, path_txt):
    img2label = {}
    with open(path_txt, 'r') as file:
        for line in file.readlines():
            img_name, label = line.split(' ')[:2]
            img2label[img_name.split('/')[-1]] = float(label)


    img_names = os.listdir(img_dir)
    images, labels = [], []
    for img_name in tqdm(img_names):
        if img2label.get(img_name) is not None:
            img = cv2.imread(os.path.join(img_dir, img_name))
            images.append(img)
            labels.append(img2label.get(img_name))
    return images, labels


if __name__ == "__main__":
    images, labels = download_images_labels('/media/main/data2/projects/glasses_classification/cropped_images',
                                            '/media/main/data2/projects/glasses_classification/data.txt')
    print(labels[:3])
    print(len(images), len(labels))