import numpy as np
from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the test data
(_, _), (test_images, _) = mnist.load_data()

# Select a random image from the test data
image_index = np.random.randint(0, test_images.shape[0])
image = test_images[image_index]

# Plot the image
plt.imshow(image, cmap='gray')
plt.show()

# Reshape the image to be fed to the model
image = image.reshape(1, 784)
image = image.astype('float32') / 255

# Load the trained model
model = load_model('mnist_model.h5')

# Make a prediction on the image
prediction = model.predict(image)
predicted_label = np.argmax(prediction)

print(f'The predicted label for the image is: {predicted_label}')
