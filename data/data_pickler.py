import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import argparse
import csv  # New import for CSV operations
import random

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--type_of_crop', type=str, required=True)
args = parser.parse_args()

data = []
X = []
y = []
class_mapping = []  # New list to store class mapping

def fast_scandir(dirname):
    categories = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(categories):
        categories.extend(fast_scandir(dirname))
    return categories

def create_data(subfolders):
    for category in subfolders:
        class_num = subfolders.index(category)
        class_name = os.path.basename(category)  # Extract class name from the path
        class_mapping.append([class_num, class_name])  # Add class num and name to mapping
        for img in tqdm(os.listdir(category)):
            try:
                img_arr = cv2.imread(os.path.join(category, img))
                new_arr = cv2.resize(img_arr, (args.img_size, args.img_size))
                data.append([new_arr, class_num])
            except Exception as e:
                print(f'Failed to process image: {img}, error: {e}')
                pass

create_data(fast_scandir(args.dataset))

random.shuffle(data)

for features, labels in data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, args.img_size, args.img_size, 3)
y = np.array(y)

# Save data to pickle files
pickle_out = open(f'X_{args.type_of_crop}.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open(f'y_{args.type_of_crop}.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

# Write class mapping to a CSV file
with open(f'{args.type_of_crop}_class_mapping.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class Number', 'Class Name'])
    for entry in class_mapping:
        writer.writerow(entry)
