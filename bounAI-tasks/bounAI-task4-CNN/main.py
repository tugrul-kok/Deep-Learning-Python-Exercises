import os
from random import shuffle
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard


class DataGenerator(Sequence):
    def __init__(self, batch_size=32, do_shuffle=True, train=True):
        self.train = train
        self.cats = os.listdir('data/Images')
        self.paths = []
        self.do_shuffle = do_shuffle
        self.batch_size = batch_size
        for cat in self.cats:
            files = os.listdir('data/Images/' + cat)
            if self.train:
                files = files[:int(len(files) * 0.8)]
            else:
                files = files[int(len(files) * 0.8):]
            for file in files:
                self.paths.append(['data/Images/' + cat + '/' + file, cat])
        if self.do_shuffle:
            shuffle(self.paths)

    def __len__(self):
        return int(len(self.paths)//self.batch_size)

    def __getitem__(self, index):
        x = np.empty((0, 128, 128, 3), dtype=np.float64)
        y = np.zeros((self.batch_size,120), dtype=np.float64)
        for j in range(self.batch_size):
            img = cv2.imread(self.paths[index*self.batch_size+j][0])
            img = cv2.resize(img, (128, 128))
            img = np.reshape(img, (1, 128, 128, 3))
            cat_index = self.cats.index(self.paths[index*self.batch_size+j][1])
            y[j,cat_index] = 1.0
            x = np.vstack((x, img))
        return x, y

    def on_epoch_end(self):
        if self.do_shuffle:
            shuffle(self.paths)


model = Sequential(
    [
        Conv2D(32, kernel_size=(3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(16, kernel_size=(3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(120, activation='softmax')
    ]
)

callbacks = [
    TensorBoard(log_dir='tb')
]
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
train_generator = DataGenerator()
val_generator = DataGenerator(train=False)
model.fit(train_generator, epochs=30, callbacks=callbacks, validation_data=val_generator)