import os
import numpy as np
from tqdm import tqdm
import cv2

class DogVSCat():

    DOG = "PetImages/Dog"
    CAT = "PetImages/Cat"

    LABELS = {DOG: 0, CAT: 1}

    def make_training_data_LeNet(self):
        image_size = 32
        training_data = []
        dog_count = 0
        cat_count = 0

        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (image_size, image_size))

                    training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.DOG:
                        dog_count += 1
                    elif label == self.CAT:
                        cat_count += 1

                except Exception as e:
                    pass

        np.random.shuffle(training_data)
        np.save("TestData/data_LeNet.npy", training_data)


    def make_training_data_AlexNet(self):
        image_size = 227
        training_data = []
        dog_count = 0
        cat_count = 0
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (image_size, image_size))

                    training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.DOG:
                        dog_count += 1
                    elif label == self.CAT:
                        cat_count += 1

                except Exception as e:
                    pass

        np.random.shuffle(training_data)
        training_data = training_data[0:5000]
        np.save("TestData/data_AlexNet.npy", training_data)

dogcat = DogVSCat()
dogcat.make_training_data_LeNet()
dogcat.make_training_data_AlexNet()