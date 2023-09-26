#This one gets frames from a live videocam feed and does not take away the background. Also make lines into colours that changes with each frame
import cv2
import numpy as np

global col2, col4, col6, col8
col2 = ()
col4 = ()
col6 = ()
col8 = ()

colors = [
    [(222, 192, 95), (255, 195, 85), (43, 100, 173), (236, 154, 190)],
    [(42, 95, 50), (81, 34, 109), (83, 57, 154), (69, 116, 154)],
    [(255, 0, 0), (0, 255, 0), (0, 0, 255), (81, 34, 109)]
]

def color_generator():
    selec = np.random.choice(len(colors))
    return colors[selec]


col2, col4, col6, col8 = color_generator()

def check_steps():
    global steps
    steps = [
        {'step': 8, 'color': col2, 'range': (153, 204)},
        {'step': 6, 'color': col4, 'range': (102, 153)},
        {'step': 4, 'color': col6, 'range': (51, 102)},
        {'step': 2, 'color': col8, 'range': (0, 51)}
    ]
check_steps()

#Open a live camera feed
cap = cv2.VideoCapture(1)
while True:
    #Get the frame
    ret, frame = cap.read()

    # Get the dimensions of the frame
    height, width, channels = frame.shape

    #-----Create canvas of same dimension as image and add image to the exact place it way in image using the coordinates
    # Define the dimensions of the white canvas
    canvas_height, canvas_width = height, width

    # Create a white canvas
    final_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # 3 for RGB channels

    # Calculate position to place the image in the center
    start_x = (canvas_width - width) // 1
    start_y = (canvas_height - height) // 1

    # Paste the color image onto the middle of the white canvas
    final_canvas[start_y:start_y+height, start_x:start_x+width] = frame
    #-----

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Apply a face detection algorithm to get the face region
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # # Get the coordinates of the face and neck
    if len(faces) > 0:

        for f in faces:

            if col2 == (): #If col2 is not empty
                col2, col4, col6, col8 = color_generator()
                check_steps()

            x, y, w, h = f
            face_neck = frame[y:y+h, x:x+w]

            # Convert the face + neck to black and white
            bw_face_neck = cv2.cvtColor(face_neck, cv2.COLOR_BGR2GRAY)

            height, width = bw_face_neck.shape

            # Create a white canvas for the lines
            white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Drawing lines based on pixel color conditions
            cntr = 4
            for s in steps:
                step = s['step']
                color = s['color']
                lower, upper = s['range']
                
                # Create a mask where the conditions are met
                mask = (bw_face_neck[::step, ::step] >= lower) & (bw_face_neck[::step, ::step] < upper)
                
                # Get the indices where mask is True and adjust the coordinates
                y1, x1 = np.where(mask)
                y1 = y1 * step
                x1 = x1 * step
                
                # Draw lines on the white_canvas
                for i, j in zip(x1, y1):
                    cv2.line(white_canvas, (i - cntr, j - cntr), (i + cntr, j + cntr), color, 1)

                cntr -=1

            # Paste the smaller image onto the middle of the larger canvas
            final_canvas[y:y+h, x:x+w] = white_canvas
        
        # Show the final image
        cv2.imshow('frame', final_canvas)

        #CHange all the cols to empty
        col2 = ()
        col4 = ()
        col6 = ()
        col8 = ()
    else:
        cv2.imshow('frame', frame)

    # Check for key events
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' or ESC key
        break
cap.release()
cv2.destroyAllWindows()