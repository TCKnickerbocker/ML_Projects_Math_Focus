import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from EMG import EMG

# Load and preprocess the image
img = io.imread('./TCF_Bank_Stadium.jpg')
img = img / 255
img_shape = img.shape
img = img.reshape(-1, 3)

# Declare k-values
k_values = [4, 8, 12]

# Create subplots for displaying enhanced images
fig, axs = plt.subplots(len(k_values), 1, figsize=(20, 10))

for i, k in enumerate(k_values):
    # Apply EM-G algorithm
    h, mu, Q = EMG(img, k, 1)
    compress = mu[np.argmax(h, axis=1), :]

    # Setup image
    axs[i].imshow(compress.reshape(*img_shape))
    axs[i].set_title('K=' + str(k))

# Save image
fig.savefig('./myStadium.png')

# Create subplots for displaying log-likelihood curves
fig, axs = plt.subplots(len(k_values), 1, figsize=(10, 5))

for i, k in enumerate(k_values):
    # Apply EM-G algorithm
    h, mu, Q = EMG(img, k, 1)

    # Plot log-likelihood curves
    axs[i].plot(range(Q.shape[0]), Q[:, 0], '+', markersize=6, color='blue')
    axs[i].plot(range(Q.shape[0]), Q[:, 1], 'o', markersize=2, color='red')
    # Setup plot details
    if i == 1:
        axs[i].set_ylabel('Complete log-likelihood', fontsize=12)
    axs[i].set_xlabel('The number of iterations', fontsize=12)
    axs[i].set_title('K=' + str(k))

# Save image
fig.savefig('./myLogLikePlot.png')
