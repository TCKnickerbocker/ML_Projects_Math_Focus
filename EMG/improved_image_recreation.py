import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from EMG import EMG  

# Load the image & preprocess it
img = io.imread('./goldy.jpg') 
w, l, c = np.shape(img)  # Get image dimensions 
img = img / 255  # Normalize image pixel values to range [0, 1]
img = np.reshape(img, (-1, 3))  # Reshape the image to a 2D array of pixels

# Create figure, apply improved EMG algorithm
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
h, mu, Q = EMG(img, 7, 1)  # k=7 used

# Compress image using mean probable components, display, save
compress = mu[np.argmax(h, axis=1), :]
axs.imshow(np.reshape(compress, (w, l, c)))  
axs.set_title('K=7') 
fig.savefig('./myGoldy.png') 
