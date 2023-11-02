from numpy import loadtxt #handle/load dataset
from keras.models import Sequential #Empty working area 
from keras.layers import Dense #Dense layer 
from keras.layers import Dropout
import keras

dataset = loadtxt('dataset.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=1)

model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


from keras.optimizers import SGD
model.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=1000, batch_size=10)

_, accuracy = model.evaluate(X_test, y_test)
print('1_Evaluate_Accuracy: %.2f' % (accuracy*100))


from sklearn.metrics import accuracy_score
import numpy as np
y_pred = model.predict(X_test)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

test=y_test
a = accuracy_score(pred,test)
print('2_AccuracyScore_Accuracy is:', a*100)


predictions = model.predict_classes(X)
for i in range(11,25):
	print('Predicted Class: %d (Original Class: %d)' % (predictions[i], y[i]))

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
