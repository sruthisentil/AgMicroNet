import os
import numpy as np
import pickle
import cv2
import random
from tqdm import tqdm
import argparse

os.getcwd()
data = []
X = []
y = []


# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--type_of_crop', type=str, required=True)
args = parser.parse_args()

def fast_scandir(dirname):
    categories= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(categories):
        categories.extend(fast_scandir(dirname))
    return categories

'''def create_data(dataset_path):
    for category in categories:
        path = os.path.join(dataset_path , category)
        class_num = categories.index(category)
        for img in tqdm(os.listdir(path)):

            try:
                img_arr = cv2.imread(os.path.join(path, img))
                new_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([new_arr, class_num])
            except cv2.error as e:
                pass'''
def create_data(subfolders):
    for category in subfolders:
        class_num = subfolders.index(category)
        for img in tqdm(os.listdir(category)):

            try:
                img_arr = cv2.imread(os.path.join(category, img))
                new_arr = cv2.resize(img_arr, (args.img_size, args.img_size))
                data.append([new_arr, class_num])
            except cv2.error as e:
                pass
create_data(fast_scandir(args.dataset))

random.shuffle(data)

for features, labels in data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, args.img_size, args.img_size, 3)
y = np.array(y)
print('Shape of X: ', X.shape)
print('Shape of y: ', y.shape)
pickle_out = open('X_' + args.type_of_crop + '.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open('y_' + args.type_of_crop + '.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
