#You but
#This one makes you select a photo to use and does not take away the background
import cv2
import numpy as np

#Open file opener
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

# Load the image
image = cv2.imread(file_path)


# Get the dimensions of the image
height, width, channels = image.shape

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# # Apply a face detection algorithm to get the face region
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

# # Get the coordinates of the face and neck
# x, y, w, h = faces[0]
# face_neck = image[y:y+h, x:x+w]

# # Convert the face + neck to black and white
# bw_face_neck = cv2.cvtColor(face_neck, cv2.COLOR_BGR2GRAY)

height, width = gray.shape

#Image canvas
image_canvas = np.ones((height + 10, width + 10), dtype=np.uint8) * 255

# Get dimensions of both images
height, width = gray.shape
canvas_height, canvas_width = image_canvas.shape

# Calculate position to place the smaller image in the center
start_x = (canvas_width - width) // 2
start_y = (canvas_height - height) // 2

# Paste the smaller image onto the middle of the larger canvas
image_canvas[start_y:start_y+height, start_x:start_x+width] = gray

#Make canvas twice as large
image_canvas = cv2.resize(image_canvas, (width*2, height*2), interpolation=cv2.INTER_AREA)

# Show the result
cv2.imshow('Result1', image_canvas)

#----------------------------
# Get the dimensions of the image
height, width = image_canvas.shape

# Create a white canvas
white_canvas = np.ones((height, width), dtype=np.uint8) * 255

# Based on bw_face_neck, use 45 degree angle lines to draw the image on the white canvas
for i in range(10, width, 8):
    for j in range(10, height, 8):
        # print(f'Pixel at ({i}, {j}): {image_canvas[j, i]}')
        pixel_col = image_canvas[j, i]
        if ((pixel_col < 204) & (pixel_col >= 179)):
            # Draw the line
            cv2.line(white_canvas, (i-5, j-5), (i + 5, j + 5), 192, 1)

#Going over the image and adding lines between the ones added before
for i in range(10, width, 7):
    for j in range(10, height, 7):
        # pixel_col = rgb_to_gray(image_canvas[j, i])
        pixel_col = image_canvas[j, i]
        if ((pixel_col < 179) & (pixel_col >= 153)):
            # Draw the line
            cv2.line(white_canvas, (i-5, j-5), (i + 5, j + 5), 166, 1)

#Going over the image and adding lines between the ones added before
for i in range(10, width, 6):
    for j in range(10, height, 6):
        # pixel_col = rgb_to_gray(image_canvas[j, i])
        pixel_col = image_canvas[j, i]
        if ((pixel_col < 153) & (pixel_col >= 128)):
            # Draw the line
            cv2.line(white_canvas, (i-4, j-4), (i + 4, j + 4), 141, 1)

#Going over the image and adding lines between the ones added before
for i in range(10, width, 5):
    for j in range(10, height, 5):
        # pixel_col = rgb_to_gray(image_canvas[j, i])
        pixel_col = image_canvas[j, i]
        if ((pixel_col < 128) & (pixel_col >= 102)):
            # Draw the line
            cv2.line(white_canvas, (i-4, j-4), (i + 4, j + 4), 114, 1)


#Going over the image and adding lines between the ones added before
for i in range(10, width, 4):
    for j in range(10, height, 4):
        # pixel_col = rgb_to_gray(image_canvas[j, i])
        pixel_col = image_canvas[j, i]
        if ((pixel_col < 102) & (pixel_col >= 77)):
            # Draw the line
            cv2.line(white_canvas, (i-3, j-3), (i + 3, j + 3), 90, 1)

#Going over the image and adding lines between the ones added before
for i in range(10, width, 3):
    for j in range(10, height, 3):
        # pixel_col = rgb_to_gray(image_canvas[j, i])
        pixel_col = image_canvas[j, i]
        if ((pixel_col < 77) & (pixel_col >= 51)):
            # Draw the line
            cv2.line(white_canvas, (i-3, j-3), (i + 3, j + 3), 74, 1)

#Going over the image and adding lines between the ones added before
for i in range(10, width, 2):
    for j in range(10, height, 2):
        # pixel_col = rgb_to_gray(image_canvas[j, i])
        pixel_col = image_canvas[j, i]
        if ((pixel_col < 51) & (pixel_col >= 26)):
            # Draw the line
            cv2.line(white_canvas, (i-2, j-2), (i + 2, j + 2), 39, 1)

#Going over the image and adding lines between the ones added before
for i in range(10, width, 1):
    for j in range(10, height, 1):
        # pixel_col = rgb_to_gray(image_canvas[j, i])
        pixel_col = image_canvas[j, i]
        if ((pixel_col < 26) & (pixel_col >= 0)):
            # Draw the line
            cv2.line(white_canvas, (i-1, j-1), (i + 1, j + 1), 13, 1)



# Show the result
cv2.imshow('Result', white_canvas)

#----------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()

