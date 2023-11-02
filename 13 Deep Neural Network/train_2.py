from numpy import loadtxt #handle/load dataset
from keras.models import Sequential #Empty working area 
from keras.layers import Dense #Dense layer 
from keras.layers import BatchNormalization

dataset = loadtxt('dataset.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]
print(x)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=5, batch_size=10)

_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

'''from sklearn.metrics import accuracy_score
import numpy as np
y_pred = model.predict(x)
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
test=y
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)'''


predictions = model.predict_classes(x)
for i in range(5,10):
	print('Predicted Class: %d (Original Class: %d)' % (predictions[i], y[i]))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
