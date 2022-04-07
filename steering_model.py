import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

import matplotlib.pyplot as plt

import cv2
import numpy as np
import csv


class SteeringDataset(data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]

        steering_angle = float(batch_samples[3])
        center_img, steering_angle_center = augment(
            batch_samples[0], steering_angle)
        left_img, steering_angle_left = augment(
            batch_samples[1], steering_angle + .4)
        right_img, steering_angle_right = augment(
            batch_samples[2], steering_angle - .4)

        center_img = self.transform(center_img)
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        return (center_img, steering_angle_center), (left_img, steering_angle_left), (right_img, steering_angle_right)

    def __len__(self):
        return len(self.samples)


class SteeringTrain:
    def __init__(self):
        self.samples = []
        self.train_len = int(0.8*len(self.samples))
        self.valid_len = len(samples) - train_len
        self.DATA_LOC = 'udacity/data/IMG/'

    def get_udacity_data(self):
        # !wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
        # !unzip data.zip -d /content/udacity/
        pass

    def populate_samples():
        samples = []
        with open('/content/udacity/data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for line in reader:
                samples.append(line)

        return samples

    def train_val_split():
        train_len = int(0.8*len(self.samples))
        valid_len = len(self.samples) - train_len
        train_samples, validation_samples = data.random_split(
            self.samples, lengths=[train_len, valid_len])

        return train_samples, validation_samples

    def augment(imgName, angle):
        name = self.DATA_LOC + imgName.split('/')[-1]
        current_image = cv2.imread(name)
        current_image = current_image[65:-25, :, :]
        if np.random.rand() < 0.5:
            current_image = cv2.flip(current_image, 1)
            angle = angle * -1.0
        return current_image, angle
