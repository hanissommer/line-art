import numpy as np
import cv2
from screeninfo import get_monitors #Need pip install screeninfo

global col2, col4, col6, col8
col2 = ()
col4 = ()
col6 = ()
col8 = ()

global steps
steps = [
    {'step': 8, 'color': col2, 'range': (153, 204)},
    {'step': 6, 'color': col4, 'range': (102, 153)},
    {'step': 4, 'color': col6, 'range': (51, 102)},
    {'step': 2, 'color': col8, 'range': (0, 51)}
]

#Color palettes -- dark to light pixels
global colors
colors = [
    [(222, 192, 95), (255, 195, 85), (43, 100, 173), (236, 154, 190)],
    [(42, 95, 50), (81, 34, 109), (83, 57, 154), (69, 116, 154)],
    [(255, 0, 0), (0, 255, 0), (0, 0, 255), (81, 34, 109)]
]

#Randomly select a color palette
def color_generator():
    global colors
    selec = np.random.choice(len(colors))
    return colors[selec]

#Initialize colors 
def initialize_colors():
    global col2, col4, col6, col8
    col2, col4, col6, col8 = color_generator()
    check_steps(col2, col4, col6, col8)

#Reinitializes the steps with new colours, if any
def check_steps(col2, col4, col6, col8):
    global steps
    steps = [
        {'step': 8, 'color': col2, 'range': (153, 204)},
        {'step': 6, 'color': col4, 'range': (102, 153)},
        {'step': 4, 'color': col6, 'range': (51, 102)},
        {'step': 2, 'color': col8, 'range': (0, 51)}
    ]

#Gets the details for the steps for line drawing
def get_steps():
    check_steps(col2, col4, col6, col8)
    return steps

# Load the pre-trained model and config files
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

#Used pretrained model to detect bodies
def detec_model_setup(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    return detections, height, width

#Clear colors
def clear_colors():
    # print("Clearing colors")
    global col2, col4, col6, col8
    col2 = ()
    col4 = ()
    col6 = ()
    col8 = ()
    initialize_colors()

#Create a white canvas
def create_canvas(height, width):
    return np.ones((height, width, 3), dtype=np.uint8) * 255


#Gets the details of the display
def get_monitor_details(monitor_number):
    # Assuming a single monitor setup
    monitor = get_monitors()[monitor_number]
    screen_width = monitor.width
    screen_height = monitor.height
    return screen_width, screen_height


#Gets the details of the display and resizes the frame to fit the display
def cv2_large(frame, screen_width, screen_height):
    # Calculate the aspect ratio preserving resize dimensions
    height, width, _ = frame.shape  # or final_canvas.shape
    aspect_ratio = width / height
    new_width = screen_width
    new_height = int(screen_width / aspect_ratio)

    if new_height > screen_height:
        new_height = screen_height
        new_width = int(screen_height * aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))  # or final_canvas
    cv2.imshow('frame', resized_frame)
    
