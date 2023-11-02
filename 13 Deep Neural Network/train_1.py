from numpy import loadtxt #handle/load dataset
from keras.models import Sequential #Empty working area 
from keras.layers import Dense #Dense layer 

dataset = loadtxt('dataset.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]
print(x)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=5, batch_size=10)

_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict_classes(x)

for i in range(5,10):
	print('Predicted Class: %d (Original Class: %d)' % (predictions[i], y[i]))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
