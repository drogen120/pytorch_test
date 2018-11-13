from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()

DATASET_FOLDER = os.path.expanduser('~') + "/Dataset/faces/"
landmarks_frame = pd.read_csv(DATASET_FOLDER + "face_landmarks.csv")
n = 65
image_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.reshape(-1,2)

print('Image name: {}'.format(image_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

def show_landmarks(image, landmarks):
	plt.imshow(image)
	plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')
	plt.pause(0)

plt.figure()
image = io.imread(os.path.join(DATASET_FOLDER, image_name))
show_landmarks(image, landmarks)
plt.show()