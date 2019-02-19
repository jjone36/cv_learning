import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from PIL import Image
img = Image.open('img.jpg')
img_arr = np.asarray(img)
img_arr.shape
plt.imshow(img_arr)

plt.imshow(img_arr[:, :, 0], cmap = 'gray')

# canvas setting
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(111)
ax.imshow(img)

#################################################
img = cv2.imread('img.jpg')
plt.imshow(img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

img_gray = cv2.imread('data/00-puppy.jpg', cv2.IMREAD_GRAYSCALE)
img_gray.shape
plt.imshow(img_gray, cmap = 'gray')

# resize
w_ratio = .5
h_ratio = .5
img_2 = cv2.resize(img, (0, 0), img, w_ratio, h_ratio)
plt.imshow(img_2)

# flip
img_2 = cv2.flip(img, 0)
plt.imshow(img_2)

# saving
cv2.imwrite('new_image.jpg', img)

########## Drawing on images
blank_img = np.zeros(shape = (512, 512, 3), dtype = np.int16)
plt.imshow(blank_img)

# rectangle
cv2.rectangle(blank_img, pt1 = (384, 10), pt2 = (510, 100), color = (0, 0, 255), thickness = 10)

# circle
cv2.circle(blank_img, center = (100, 100), radius = 50, color = (255, 0, 0), thickness = 8)


# line
cv2.line(blank_img, pt1 = (0, 0), plt = (512, 512), color = (102, 221, 103), thickness = 5)

# polygons
vertices = np.array([pt1, pt2, pt3], dtype = np.int32)
pts = vertices.reshape((-1, 1, 2))
cv2.polylines(blank_img, [pts], isClosed = True, color = (255, 0, 0), thickness = 5)

# fillpoly
vertices = np.array([pt1, pt2, pt3], dtype = np.int32)
pts = vertices.reshape((-1, 1, 2))
cv2.fillPoly(img_2, [pts], color = (0, 0, 255))

# font
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img, text = 'Hello', org = (10, 500), fontFace = font, fontScale = 4,
            color = (255, 255, 255), thickness = 3, lineType = cv2.LINE_AA)



