#Original ish - 
#Asks the user to choose a photo, make that into lines, then paste it back on the og image using same coords..
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


#-----Create canvas of same dimension as image and add image to the exact place it way in image using the coordinates
# Define the dimensions of the white canvas
canvas_height, canvas_width = height, width

# Create a white canvas
final_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # 3 for RGB channels

# Calculate position to place the image in the center
start_x = (canvas_width - width) // 1
start_y = (canvas_height - height) // 1

# Paste the color image onto the middle of the white canvas
final_canvas[start_y:start_y+height, start_x:start_x+width] = image
#-----


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply a face detection algorithm to get the face region
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# # Get the coordinates of the face and neck
# x, y, w, h = faces[0]

for f in faces:
    x, y, w, h = f
    face_neck = image[y:y+h, x:x+w]

    # Convert the face + neck to black and white
    bw_face_neck = cv2.cvtColor(face_neck, cv2.COLOR_BGR2GRAY)

    #Show the result
    # cv2.imshow('Result', bw_face_neck)

    height, width = bw_face_neck.shape


    # Create a white canvas for the lines
    white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Add lines for lighter shades of gray
    for i in range(0, width, 8):
        for j in range(0, height, 8):
            # print(f'Pixel at ({i}, {j}): {image_canvas[j, i]}')
            pixel_col = bw_face_neck[j, i]
            if ((pixel_col < 204) & (pixel_col >= 153)):
                # Draw the line
                cv2.line(white_canvas, (i-4, j-4), (i + 4, j + 4), 0, 1)

    #Add lines for even darker shades of gray
    for i in range(0, width, 6):
        for j in range(0, height, 6):
            # pixel_col = rgb_to_gray(image_canvas[j, i])
            pixel_col = bw_face_neck[j, i]
            if ((pixel_col < 153) & (pixel_col >= 102)):
                # Draw the line
                cv2.line(white_canvas, (i-3, j-3), (i + 3, j + 3), 0, 1)

    #Add lines for little more darker shades of gray
    for i in range(0, width, 4):
        for j in range(0, height, 4):
            # pixel_col = rgb_to_gray(image_canvas[j, i])
            pixel_col = bw_face_neck[j, i]
            if ((pixel_col < 102) & (pixel_col >= 51)):
                # Draw the line
                cv2.line(white_canvas, (i-2, j-2), (i + 2, j + 2), 0, 1)

    #Same as above but for even more darker shades of gray
    for i in range(0, width, 2):
        for j in range(0, height, 2):
            # pixel_col = rgb_to_gray(image_canvas[j, i])
            pixel_col = bw_face_neck[j, i]
            if ((pixel_col < 51) & (pixel_col >= 0)):
                # Draw the line
                cv2.line(white_canvas, (i-1, j-1), (i + 1, j + 1), 0, 1)


    # Show the result
    # cv2.imshow('Result11', white_canvas)

    # white_canvas_3channel = cv2.merge((white_canvas, white_canvas, white_canvas))

    # Paste the smaller image onto the middle of the larger canvas
    final_canvas[y:y+h, x:x+w] = white_canvas

# Show the result
cv2.imshow('Result111', final_canvas)


#----------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()

