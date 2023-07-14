import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras.optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, InputLayer
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pynput

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')

tensorflow.random.set_seed(42)
image_path = "DataSet/images/"

image_labels = ["Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk", "Hydrant", "Motorcycle", "Other", "Palm", "Stair", "Traffic Light"]

img_list = []
label_list = []

for label in image_labels:
    for img_file in os.listdir(image_path+label):
        img_list.append(image_path+label+'/'+img_file)
        label_list.append(label)

df = pd.DataFrame({'img':img_list, 'label':label_list})

print(df['label'].value_counts())

print(plt.imread(df['img'][0]).shape)

df_labels = {"Bicycle": 0, "Bridge": 1, "Bus": 2, "Car": 3, "Chimney": 4, "Crosswalk": 5, "Hydrant": 6, "Motorcycle": 7, "Other": 8, "Palm": 9, "Stair": 10, "Traffic Light": 11}

df['encode_label'] = df['label'].map(df_labels)

X = []
for img in df['img']:
    img = cv2.imread(str(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = inverted_img = cv2.bitwise_not(img)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # img = augment_function(img)
    img = cv2.resize(img, (96, 96))
    img = img/255
    final_img = cv2.merge((img, img, img))
    X.append(final_img)

y = df['encode_label']

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, random_state=42)

it_train = datagen.flow(np.array(X_train), y_train.to_numpy(), batch_size=128)

base_model = VGG16(input_shape=(96,96,3), include_top=False, weights='imagenet')
print(base_model.summary())

model = Sequential()
model.add(InputLayer(input_shape=(96,96,3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(len(df_labels), activation='softmax'))

model.compile(
  optimizer=tensorflow.keras.optimizers.Adam(lr=0.00002),
  loss='sparse_categorical_crossentropy',
  metrics=['acc'])

print(model.summary())
history = model.fit(it_train, epochs=9, validation_data=(np.array(X_val), y_val.to_numpy()), batch_size=4)


loss, accuracy = model.evaluate(np.array(X_test), y_test.to_numpy())
print("Accuracy: ", accuracy)

dwld = input("Do you want to save this model: ")

if dwld.upper() == "Y":
    model.save("my_image_rec_model.h5")
    print("Downloaded Successfully")

plt.plot(history.history['acc'], marker='o')
plt.plot(history.history['val_acc'], marker='o')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()

plt.plot(history.history['loss'], marker='o')
plt.plot(history.history['val_loss'], marker='o')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# load the image
img = cv2.imread("Car2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = inverted_img = cv2.bitwise_not(img)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
img = cv2.resize(img, (96, 96)) / 255.0
final_img = cv2.merge((img, img, img))

# make a prediction
pred = model.predict(np.array([final_img]))

# get the predicted label
pred_label = image_labels[np.argmax(pred)]

print("The model predicts that the image is a", pred_label)