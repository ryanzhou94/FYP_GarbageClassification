import os
import numpy as np
from tqdm import tqdm
import cv2

class Garbage():

    CARDBOARD = "dataset-resized/cardboard"
    GLASS = "dataset-resized/glass"
    METAL = "dataset-resized/metal"
    PAPER = "dataset-resized/paper"
    PLASTIC = "dataset-resized/plastic"
    TRASH = "dataset-resized/trash"
    LABELS = {CARDBOARD: 0, GLASS: 1, METAL: 2, PAPER: 3, PLASTIC: 4, TRASH: 5}

    def make_training_data_LeNet(self):
        image_size = 32
        training_data = []
        cardboard_count = 0
        glass_count = 0
        metal_count = 0
        paper_count = 0
        plastic_count = 0
        trash_count = 0
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (image_size, image_size))

                    training_data.append([np.array(img), np.eye(6)[self.LABELS[label]]])

                    if label == self.CARDBOARD:
                        cardboard_count += 1
                    elif label == self.GLASS:
                        glass_count += 1
                    elif label == self.METAL:
                        metal_count += 1
                    elif label == self.PAPER:
                        paper_count += 1
                    elif label == self.PLASTIC:
                        plastic_count += 1
                    elif label == self.TRASH:
                        trash_count += 1

                except Exception as e:
                    pass

        np.random.shuffle(training_data)
        np.save("TrainingData/training_garbage_data_LeNet.npy", training_data)

    def make_training_data_AlexNet(self):
        image_size = 227
        training_data = []
        cardboard_count = 0
        glass_count = 0
        metal_count = 0
        paper_count = 0
        plastic_count = 0
        trash_count = 0
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (image_size, image_size))

                    training_data.append([np.array(img), np.eye(6)[self.LABELS[label]]])

                    if label == self.CARDBOARD:
                        cardboard_count += 1
                    elif label == self.GLASS:
                        glass_count += 1
                    elif label == self.METAL:
                        metal_count += 1
                    elif label == self.PAPER:
                        paper_count += 1
                    elif label == self.PLASTIC:
                        plastic_count += 1
                    elif label == self.TRASH:
                        trash_count += 1

                except Exception as e:
                    pass

        np.random.shuffle(training_data)
        np.save("TrainingData/training_garbage_data_AlexNet.npy", training_data)

garbage = Garbage()
garbage.make_training_data_LeNet()
garbage.make_training_data_AlexNet()