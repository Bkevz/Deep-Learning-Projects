#importing Libraries
from keras.datasets import mnist #download the digit dataset
from keras.utils import to_categorical #
from keras.models import Sequential #arranging the layer in sequential order
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense #
import matplotlib.pyplot as plt #visualizing
import time
#Load Dataset
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

#Reshape dataset to have a single channel
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

plt.axis('off')
plt.imshow(X_test[45],cmap='gray')
plt.show()

#Normalizing the pixel values
X_train=X_train/255
X_test=X_test/255

#One hot encode target values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Neural Network
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(100,activation='relu')) #FCL
model.add(Dense(10,activation='softmax')) #Output Layer

#Compile Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

startTime = time.time()
#Train the Model
history=model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs=10,batch_size=32)
stopTime = time.time()
diff = stopTime-startTime
print(diff)
#Evaluate the Model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

#Graphical Representation of Accuracy & Loss Graph
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

#Save Model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
