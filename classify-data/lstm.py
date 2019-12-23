from __future__ import division
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from collections import deque
from keras.models import load_model
from keras.utils import normalize, to_categorical

# fix random seed for reproducibility
np.random.seed(7)

TIME_STEPS = 150

def ip_to_seq(csv_list):
    for i in range(len(csv_list)):
        if(i == 1 or i == 2):
          ip_list = csv_list[i].split(".")
          for j in range(len(ip_list)):
            numberOfZeros= 3-len(ip_list[j])
            ip_list[j] = str(numberOfZeros * '0')+ip_list[j]
            ip_seq = "".join(ip_list)
            csv_list[i] = ip_seq
        if(i == 4):
          if csv_list[4] == "64\n":
              csv_list[4] = "0\n"
          else:
              csv_list[4] = "1\n"

def ip_to_seq_no_label(csv_list):
    for i in range(len(csv_list)):
        if(i == 1 or i == 2):
          ip_list = csv_list[i].split(".")
          for j in range(len(ip_list)):
            numberOfZeros= 3-len(ip_list[j])
            ip_list[j] = str(numberOfZeros * '0')+ip_list[j]
            ip_seq = "".join(ip_list)
            csv_list[i] = ip_seq

class lstm_model:

    def __init__(self):
        self.ITERATIONS = 50
        self.packet_count = 0
        self.epochs = 0
        self.correct_predictions = 0
        self.sequence = []
        self.model = load_model("/home/ubuntu/Desktop/UDP-Attack-Detection-in-SDN-using-LSTM/classify-data/lstm_model.h5")
        self.model._make_predict_function()

    def classify(self, time, src_ip, dst_ip, protocol, label):
        if self.packet_count < 5*TIME_STEPS:
            self.packet_count += 1
            packet = [time, src_ip, dst_ip, protocol, label]
            ip_to_seq_no_label(packet)
            self.sequence.append(packet)
        else:
            batches, labels = create_batches(self.sequence)
            batches = np.array(batches, dtype='f')
            labels = np.array(labels, dtype='f')
            # prediction = self.model.predict(batches)
            prediction = self.model.evaluate(batches, labels)
            print(prediction)
            if prediction == 1:
                print("*** DDOS Detected ***")
            if prediction == labels[0]:
                self.correct_predictions += 1
            self.epochs += 1
            if self.epochs >= self.ITERATIONS:
                accuracy = 100 * float(self.correct_predictions)/self.epochs
                print("The accuracy is", round(accuracy, 2), "%")
            self.sequence = []
            self.packet_count = 0

# def create_badges(data):
#     batches = []
#     labels = []
#     current_batch = []
#     labels_count = 0
#     packet_count = 0
#     counter= 0
#     print(len(data))
#     for row in data:
#         current_batch.append(row[:4])
#         packet_count += 1
#         labels_count += int(row[4])
#         counter += 1
#         if counter == TIME_STEPS:
#             batches.append(current_batch)
#             fraction = float(labels_count)/packet_count
#             if fraction > 0.5:
#                 print("label", 1)
#                 labels.append(1)
#             else:
#                 print("label", 0)
#                 labels.append(0)

#     return batches, labels

def create_batches(data):
    
    batches = []
    labels = []
    current_batch = []
    labels_count = 0
    counter= 0
    for row in data:
        current_batch.append(row[:4])
        labels_count += int(row[4])
        counter += 1
        if counter == TIME_STEPS:
            batches.append(current_batch)
            fraction = float(labels_count)/counter
            if fraction > 0.5:
                print("label", 1)
                labels.append(1)
            else:
                print("label", 0)
                labels.append(0)
            current_batch = []
            labels_count = 0
            counter = 0
       
    return batches, labels

def main():
    """
    train the model from generated training data in generate-data folder
    """
    data = np.genfromtxt('../generate-data/clean_test.csv', delimiter=',')
    data = np.ndarray.tolist(data)
    batches, labels = create_batches(data)
    
    batches = np.array(batches, dtype='f')
    print(batches.shape)
    labels = np.array(labels,dtype='f')
    # print(batches[:10])
    # num = 0
    # batches.reshape(len(batches),1,5)
    
    print(batches.shape)
    print(labels.shape)
    # data_seq = data[:, 0:4]
    # labels = data[:, 4]
    print(batches[0])
    
    X_train, X_test, Y_train, Y_test = train_test_split(batches,labels, test_size = 0.2, random_state = 4 )

    model = Sequential()
    model.add(LSTM((1), unroll=True, batch_input_shape=(None, TIME_STEPS, 4), return_sequences= False, dropout=0.1, recurrent_dropout=0.1))
    # model.add(LSTM((1),  return_sequences=True))
    # model.add(LSTM((1),  return_sequences=False))
    # model.add(Dense(1))
    # model.add(Dense(1, activation='linear'))
    model.add(BatchNormalization(momentum = 0.01 ))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # fit model
    print(model.summary())
    # print(X_train)
    model.fit(X_train,  Y_train, epochs=10, validation_data=(X_test, Y_test), shuffle=True)

    # model.fit(batches,  labels, epochs=5)

    prediction = model.predict(X_test)
    print(prediction)
    # # # save model to single files
    model.save('lstm_model.h5')

if __name__ == "__main__":

    main()