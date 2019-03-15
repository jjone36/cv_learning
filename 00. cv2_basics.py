import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Importing the images
def load_img(filename):
    img = cv2.imread(filename).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Canvas setting
def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')


from PIL import Image
img = Image.open('img.jpg')
img_arr = np.asarray(img)
img_arr.shape
plt.imshow(img_arr)

plt.imshow(img_arr[:, :, 0], cmap = 'gray')

#################################################
img = cv2.imread('img.jpg')
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

########## Drawing
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

############################################################
############################## Blending
# blending with the same size
blended = cv2.addWeighted(src1 = img_1, alpha = .5, src2 = img_2, beta = .5, gamma = 10)

# overlaying with the different size
large_img = img_1
small_img = img_2

x_start = 0
x_end = x_start + small_img.shape[1]
y_start = 0
y_end = y_start + small_img.shape[0]

large_img[y_start:y_end, x_start:x_end] = small_img


############################## Threshold
ret, thresh_1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # THRESH_BINARY_INV, THRESH_TRUNC
plt.imshow(thresh_1, cmap = 'gray')

thresh_2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 8)


############################## Bluring & Smoothing
# gamma correction
img_2 = np.power(img, gamma)
show_img(img_2)

# convolution with 2d kernel
kernel = np.ones(shape = (5, 5), dtype = np.float32) / 25
img_2 = cv2.filter2D(img, -1, kernel)

# cv2 builtin
cv2.blur(img, ksize = (5, 5))
cv2.GaussianBlur(img, ksize = (5, 5), 10)
cv2.medianBlur(img, 5)
cv2.bilateralfilter(img, 9, 75, 75)


############################## Morphological Operator
kernel = np.ones((5, 5), np.uint8)

# erosion (eroding away the boundary)
img_2 = cv2.erode(img, kernel, iterations = 3)
plt.imshow(img_2)

# opening (removing background noise)
img_2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# closing (removing foreground noise)
img_2 = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)

# diff between dilation & erosion
img_2 = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)


############################## Gradients
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)
laplacian = cv2.Laplacian(img, cv2.CV_64F)


############################## Color Histogram
hist_values = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_values)

color = ('b','g','r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.plot(histr, color = col)
    plt.xlim([0,256])
plt.show()

# Histogram Equalization
img_2 = cv2.equalizeHist(img)


############################## Object Detection
# Harris Corner Detection
gray = np.float32(gray_img)
dst = cv2.cornerHarris(src = gray, blockSize = 2, ksize = 3, k = .04)
dst = cv2.dilate(dst, None)
img[dst > .01*dst.max()] = [255, 0, 0]
plt.imshow(img)

# Shi-Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(gray_img, maxCorners = 5, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y), 3, (255, 0, 0), -1)
plt.imshow(img)

# Canny Edge Detection
edges = cv2.Canny(image=img, threshold1=127, threshold2=127)

med_val = np.median(img)
lower = int(max(0, 0.7* med_val))
upper = int(min(255,1.3 * med_val))
blurred_img = cv2.blur(img,ksize=(5,5))
edges = cv2.Canny(image=blurred_img, threshold1=lower , threshold2=upper+50)
plt.imshow(edges)

# Grid Detection
found, corners = cv2.findChessboardCorners(img, (7,7))
found, corners = cv2.findCirclesGrid(img, (7,7), cv2.CALIB_CB_SYMMETRIC_GRID)

found
img_copy = img.copy()
cv2.drawChessboardCorners(img_copy, (7, 7), corners, found)

# Contour Detection
image, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

in_contours = np.zeros(image.shape)
ex_contours = np.zeros(image.shape)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        # draw the external contours
        cv2.drawContours(ex_contours, contours, i, (255, 0, 0), -1)
    elif:
        # Draw the internal contours
        cv2.drawContours(in_contours, contours, i, (255, 0, 0), -1)

plt.imshow(in_contours)



# Haar Cascades
cascade = cv2.CascadeClassifier(filepath)

def detect(img):
    img_2 = img.copy()
    rects = cascade.detectMultiScale(img_2, scaleFactor = 1.2, minNeighbors = 3)
    for (x, y, w, h) in rects:
        cv2.rectangle(img_2, (x, y), (x+w, y+h), (255, 0, 0), 3)
    return img_2
