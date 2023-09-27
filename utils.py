import numpy as np
import cv2
from screeninfo import get_monitors

class Utils:
    def __init__(self):
        self.colors = [
            [(222, 192, 95), (255, 195, 85), (43, 100, 173), (236, 154, 190)],
            [(42, 95, 50), (81, 34, 109), (83, 57, 154), (69, 116, 154)],
            [(255, 0, 0), (0, 255, 0), (0, 0, 255), (81, 34, 109)]
        ]
        self.initialize_colors()
        self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

    def color_generator(self):
        selec = np.random.choice(len(self.colors))
        return self.colors[selec]

    def initialize_colors(self):
        self.col2, self.col4, self.col6, self.col8 = self.color_generator()
        self.check_steps()

    def check_steps(self):
        self.steps = [
            {'step': 8, 'color': self.col2, 'range': (153, 204)},
            {'step': 6, 'color': self.col4, 'range': (102, 153)},
            {'step': 4, 'color': self.col6, 'range': (51, 102)},
            {'step': 2, 'color': self.col8, 'range': (0, 51)}
        ]

    #Gets the details for the steps for line drawing
    def get_steps(self):
        self.check_steps()
        return self.steps

    #Used pretrained model to detect bodies
    def detec_model_setup(self, frame):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections, height, width

    #Clear colors
    def clear_colors(self):
        # print("Clearing colors")
        self.col2 = ()
        self.col4 = ()
        self.col6 = ()
        self.col8 = ()
        self.initialize_colors()

    #Create a white canvas
    def create_canvas(self, height, width):
        return np.ones((height, width, 3), dtype=np.uint8) * 255


    #Gets the details of the display
    def get_monitor_details(self, monitor_number):
        # Assuming a single monitor setup
        monitor = get_monitors()[monitor_number]
        screen_width = monitor.width
        screen_height = monitor.height
        return screen_width, screen_height


    #Gets the details of the display and resizes the frame to fit the display
    def cv2_large(self, frame, screen_width, screen_height):
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
        

    #Checks if the detected face is large enough
    def face_large_enough(self, faces, frame, height, width):
        if len(faces) == 0:
            return False
        
        for f in faces:

            x, y, w, h = f
            face_neck = frame[y:y+h, x:x+w]

            # Convert the face + neck to black and white
            bw_face_neck = cv2.cvtColor(face_neck, cv2.COLOR_BGR2GRAY)

            height1, width1 = bw_face_neck.shape

            #If face_neck takes up a third of the frame, return yes
            if (width1*height1) > (height*width)/20:
                return True
            else:
                return False

    # #A function that checks if the body detection box has moved significantly since the last frame          
    def body_moved(self, detections, height, width, prev_box, prev_detections):
        pixel_threshold = 10  # for example, assume 10 pixels correspond to a few millimeters
        
        if prev_detections is not None:
            for f in range(detections.shape[2]):
                if detections[0, 0, f, 2] > 0.6:
                    box = detections[0, 0, f, 3:7] * np.array([width, height, width, height])
                    # prev_box = prev_detections[0, 0, f, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    (prev_startX, prev_startY, prev_endX, prev_endY) = prev_box.astype("int")
                    
                    # Calculating the differences in coordinates
                    diff_startX = abs(startX - prev_startX)
                    # diff_startY = abs(startY - prev_startY)
                    diff_endX = abs(endX - prev_endX)
                    # diff_endY = abs(endY - prev_endY)
                    
                    # Check whether the differences in any of the coordinates exceed the pixel threshold
                    if (diff_startX > pixel_threshold  or 
                        diff_endX > pixel_threshold):
                        return True
                
        return False  # Default to return False if the conditions above are not met
