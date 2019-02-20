import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importing the images
def load_img(filename):
    img = cv2.imread(filename).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 


# Canvas setting
def show_img(img, cmap = None):
    fig = plt.figure(figsize = (15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = cmap)
    


