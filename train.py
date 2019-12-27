import numpy as np
import csv
import cv2

from keras.models import Sequential
from keras.layers import Dense, Flatten

def load_data():
    lines = []
    with open('Data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'Data/IMG/'+filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

def train(X_train, y_train):
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
    model.save('model.h5')
if __name__ == "__main__":
    X_train, y_train = load_data()
    train(X_train, y_train)