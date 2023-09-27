import cv2
from cv2 import CascadeClassifier, COLOR_BGR2GRAY, line, imshow, waitKey, destroyAllWindows  # More specific imports
import numpy as np
from utils import Utils

class DynamicRunner:
    def __init__(self):
        self.utils = Utils()
        self.screen_width, self.screen_height = self.utils.get_monitor_details(1)
        self.cap = cv2.VideoCapture(1)
        self.col_clear_check = True
        self.prev_box = None
        self.prev_detection = None
        self.valid_face_takeover = False
        self.face_cascade = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        
    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def draw_lines(self, bw_face_neck, white_canvas, step, color, lower, upper):
        mask = (bw_face_neck[::step, ::step] >= lower) & (bw_face_neck[::step, ::step] < upper)
        y, x = np.where(mask)
        y = y * step
        x = x * step
        for i, j in zip(x, y):
            line(white_canvas, (i - step // 2, j - step // 2), (i + step // 2, j + step // 2), color, 1)
    

    def draw_body_detections(self, frame, col_change):
        # self.final_canvas = self.utils.create_canvas(height, width)
        detections, height, width = self.utils.detec_model_setup(frame)

        if col_change is True:
            self.utils.initialize_colors()

        final_canvas = self.utils.create_canvas(height, width)

        # col_clear_check = True

        # Use a flag to check whether a valid human detection has occurred in the current frame
        valid_detection = False

        if detections is not None:

            for f in range(detections.shape[2]):
                confidence = detections[0, 0, f, 2]

                if confidence > 0.6:
                    box = detections[0, 0, f, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")

                    if startX < endX and startY < endY:  # This checks if the bounding box is valid.
                        face_neck = frame[startY:endY, startX:endX]

                        if face_neck.size > 0:  # This checks if the sliced face_neck is not empty.
                            bw_face_neck = cv2.cvtColor(face_neck, cv2.COLOR_BGR2GRAY)
                            valid_detection = True  # Set the flag to True as we have a valid detection

                            new_height, new_width = bw_face_neck.shape
                            white_canvas = self.utils.create_canvas(new_height, new_width)

                            # Drawing lines based on pixel color conditions
                            for s in self.utils.get_steps():
                                step = s['step']
                                color = s['color']
                                lower, upper = s['range']
                                
                                self.draw_lines(bw_face_neck, white_canvas, step, color, lower, upper)

                            # After processing all steps, assign white_canvas to the appropriate location on final_canvas
                            if white_canvas.shape == final_canvas[startY:endY, startX:endX].shape:
                                final_canvas[startY:endY, startX:endX] = white_canvas
                            else:
                                self.utils.cv2_large(frame, self.screen_width, self.screen_height)
                                print(f"Shape mismatch: white_canvas: {white_canvas.shape}, final_canvas slice: {final_canvas[startY:endY, startX:endX].shape}")
                        else:
                            self.utils.cv2_large(frame, self.screen_width, self.screen_height)
                    self.prev_box = box
                    self.prev_detection = detections

            if valid_detection:
                self.col_clear_check = False
                # self.prev_detection = detections
                self.utils.cv2_large(final_canvas, self.screen_width, self.screen_height)
                
            else:
                self.utils.cv2_large(frame, self.screen_width, self.screen_height)  # If no valid detection, display the original frame
                if self.col_clear_check == False:
                    self.utils.clear_colors()
                    self.col_clear_check = True

    
    def draw_face_detections(self, frame, col_change):
        # Get the dimensions of the frame
        height, width, channels = frame.shape

        if col_change is True:
            self.utils.initialize_colors()

        #-----Create canvas of same dimension as image and add image to the exact place it way in image using the coordinates
        # Define the dimensions of the white canvas
        canvas_height, canvas_width = height, width

        # Create a white canvas
        final_canvas = self.utils.create_canvas(canvas_height, canvas_width)

        # Calculate position to place the image in the center
        start_x = (canvas_width - width) // 1
        start_y = (canvas_height - height) // 1

        # Paste the color image onto the middle of the white canvas
        final_canvas[start_y:start_y+height, start_x:start_x+width] = frame
        #-----

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        col_clear_check1 = True


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

                height1, width1 = bw_face_neck.shape

                # Create a white canvas for the lines
                white_canvas = np.ones((height1, width1, 3), dtype=np.uint8) * 255

                # Drawing lines based on pixel color conditions
                for s in self.utils.get_steps():
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
            # cv2.imshow('frame', final_canvas)
            self.utils.cv2_large(final_canvas, self.screen_width, self.screen_height)
            col_clear_check1 = False

        else:
            cv2.imshow('frame', frame)
            if col_clear_check1 == False:
                self.utils.clear_colors()
                col_clear_check1 = True

    def process_frame(self, frame):
        # Logic for processing each frame
        # break this method into more methods if needed
        detections, height, width = self.utils.detec_model_setup(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a face detection algorithm to get the face region
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        col_change = self.utils.body_moved(detections, height, width, self.prev_box, self.prev_detection)

        self.valid_face_takeover = self.utils.face_large_enough(faces, frame, height, width)

        if self.valid_face_takeover is False:
            self.draw_body_detections(frame, col_change)
        else:
            self.draw_face_detections(frame, col_change)


        # self.prev_detection = detections

    
    def run(self):

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.process_frame(frame)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
        self.release_resources()

def actual_run():
    runner = DynamicRunner()
    runner.run()

if __name__ == "__main__":
    actual_run()
