import os, os.path
import cv2
import random
from glob import glob
import numpy as np

print(len(os.listdir('data/train_0')))
print(len(os.listdir('data/train_25')))
print(len(os.listdir('data/train_50')))
print(len(os.listdir('data/train_75')))
print(len(os.listdir('data/train_100')))

testing_faces_dir = 'data/testing_faces'
testing_nonfaces_dir = 'data/testing_nonfaces'

# Get the names of the first n testing faces
face_filenames = sorted(glob(os.path.join(testing_faces_dir, '*.jpg')))
race = face_filenames[0].split('_')[2]

print(race)
