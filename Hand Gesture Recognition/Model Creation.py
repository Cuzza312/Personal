import tensorflow as tf
from tensorflow.keras import layers

# Prepare the data
train_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.1, rotation_range=4, width_shift_range=0.15, height_shift_range = 0.2, shear_range=0.3, fill_mode='nearest')
train_generator = train_data.flow_from_directory(directory='Data/train/', target_size=(64, 64),
                                                 batch_size=256, class_mode='categorical', subset='training')
test_generator = train_data.flow_from_directory(directory='Data/test/', target_size=(64, 64),
                                                batch_size=64, class_mode='categorical', subset='validation')

# Define the model architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

# Make predictions on new data
new_data = tf.keras.preprocessing.image.load_img('Stop.jpg', target_size=(64, 64))
new_data = tf.keras.preprocessing.image.img_to_array(new_data)
new_data = new_data / 255.
new_data = tf.expand_dims(new_data, axis=0)
prediction = model.predict(new_data)

# Get the predicted label
predicted_class = tf.argmax(prediction, axis=1)
print(f'Predicted class: {predicted_class[0]}')

model.save("my_hand_rec_model3.h5")
