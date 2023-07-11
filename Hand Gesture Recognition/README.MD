# Hand Gesture Recognition

This was an end of unit project for school where I had to create an AI that recognises hand gestures. It was designed to be further added onto so it could be used to control a robot

The data for training the AI was gathered from a Kaggle source and my own data was added for diversity


The model itself is a multilayered CNN. The project works by using OpenCV to check for hands inside the camera frame. If detected it takes a grab, gets the position of the hand and takes a cropped image of it. The image is then converted to grey-scale, inverted and thresholding is applied to it. The image is then saved and preproccesed to be the same size as the model input then fed into the model and returned as a 0-3. The return is turned into a label and printed. The possibility of this return is endless as you can run code depending on what number is returned.
