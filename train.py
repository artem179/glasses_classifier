import cv2
import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from sklearn.model_selection import train_test_split
from utils.transformation import data_transforms
from utils.data import EyeGlassesDataset
from nets.tinyVGG import SimpleVGG
from tensorboardX import SummaryWriter
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler

writer = SummaryWriter()


def train_model(model, optimizer, scheduler, criterion, num_epochs=25):
    best_loss = 1e10
    train_loss = []
    val_loss = []
    g_t = 0
    g_v = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        correct = 0
        total = 0
        train_total = 0
        train_correct = 0
        train_loss = 0.
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.view(-1).to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        train_loss += loss.item()
                        train_total += labels.size(0)
                        x = vutils.make_grid(inputs[:32], normalize=True, scale_each=True)
                        writer.add_image('Augmented_images', x, g_t)
                        writer.add_scalar('data/train_loss', loss.item() / labels.size(0), g_t)
                        g_t += 1
                        _, predicted = torch.max(outputs.data, 1)
                        train_correct += (predicted == labels).sum().item()
                        loss.backward()
                        optimizer.step()
                    else:
                        writer.add_scalar('data/val_loss', loss.item() / labels.size(0), g_v)
                        g_v += 1
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                epoch_samples += inputs.size(0)
            if phase == 'train':
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        time_elapsed = time.time() - since
        writer.add_scalar('data/validation_accuracy', 100 * correct / total, epoch)
        writer.add_scalar('data/train_accuracy', 100 * train_correct / train_total, epoch)
        writer.export_scalars_to_json("./all_scalars.json")
    return model


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


def calculate_stats(dataset):
    images = [(np.array(image)/255.).reshape(3, -1) for image in dataset]
    images = np.concatenate(images, 1)
    mean = images.mean(1)
    std = images.std(1)
    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    images, labels = download_images_labels('cropped_images', 'data.txt')

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    X = {'train': X_train, 'val': X_test}
    del X_train, X_test
    y = {'train': y_train, 'val': y_test}
    del y_train, y_test

    b_size = 256
    image_datasets = {x: EyeGlassesDataset(X[x], y[x], data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=b_size,
                                 shuffle=True, num_workers=8)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(y[x]) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleVGG(2)
    model.to(device)

    optimizer_ft = optim.SGD(model.parameters(), lr=2e-4, momentum=0.95)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=200, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model = train_model(model, optimizer_ft, exp_lr_scheduler, criterion, num_epochs=600)
    torch.save(model.state_dict(), 'trained_modelparams/best_weights.pth')
