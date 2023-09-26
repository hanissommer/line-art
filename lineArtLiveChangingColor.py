import cv2
import numpy as np
from utils import clear_colors, create_canvas, initialize_colors, get_steps


def run_lalcc():
    initialize_colors()
    col_clear_check = True

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
        final_canvas = create_canvas(canvas_height, canvas_width)

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
                x, y, w, h = f
                face_neck = frame[y:y+h, x:x+w]

                # Convert the face + neck to black and white
                bw_face_neck = cv2.cvtColor(face_neck, cv2.COLOR_BGR2GRAY)

                height, width = bw_face_neck.shape

                # Create a white canvas for the lines
                white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

                # Drawing lines based on pixel color conditions
                for s in get_steps():
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
                        cv2.line(white_canvas, (i - step//2, j - step//2), (i + step//2, j + step//2), color, 1)

                # Paste the smaller image onto the middle of the larger canvas
                final_canvas[y:y+h, x:x+w] = white_canvas
            
            # Show the final image
            cv2.imshow('frame', final_canvas)
            col_clear_check = False
            
        else:
            cv2.imshow('frame', frame)
            if col_clear_check == False:
                clear_colors()
                col_clear_check = True

        # Check for key events
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' or ESC key
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_lalcc()