from skimage import io
import cv2 as cv
from google.colab.patches import cv2_imshow
import numpy as np

image_1 = io.imread("SOCOFing/Real/1__M_Left_index_finger.BMP")
print(image_1.shape)
image_2 = io.imread("SOCOFing/Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP")
print(image_2.shape)
image_3 = io.imread("SOCOFing/Real/5__M_Right_little_finger.BMP")
print(image_3.shape)

image_1 = cv.cvtColor(image_1, cv.COLOR_BGR2GRAY)
image_3 = cv.cvtColor(image_3, cv.COLOR_BGR2GRAY)
final_frame = np.concatenate((image_1, image_2, image_3), axis=1)
cv2_imshow(final_frame)
