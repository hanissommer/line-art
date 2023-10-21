#Switched to face detection with increased resolution when you get close to the camera
import cv2
from cv2 import CascadeClassifier, COLOR_BGR2GRAY, line, imshow, waitKey, destroyAllWindows  # More specific imports
import numpy as np
from utils_2 import Utils

class DynamicRunner:
    def __init__(self):
        self.utils = Utils()
        # print(self.utils.screen_height, self.utils.screen_width)
        self.col_clear_check = True
        self.prev_box = None
        self.prev_detection = None
        self.curr_f = None
        self.valid_face_takeover = False
        self.past_frame = None
        self.past_human_detection = None
        self.face_cascade = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        
    def release_resources(self):
        self.utils.cap.release()
        cv2.destroyAllWindows()


    def draw_lines(self, bw_face_neck, white_canvas, s):
        step = s['step']
        color = s['color']
        lower, upper = s['range']
        mask = (bw_face_neck[::step, ::step] >= lower) & (bw_face_neck[::step, ::step] < upper)
        y, x = np.where(mask)
        y = y * step
        x = x * step
        for i, j in zip(x, y):
            line(white_canvas, (i - step // 2, j - step // 2), (i + step // 2, j + step // 2), color, 1)
    

    def draw_body_detections(self, frame):
        detections, height, width = self.utils.detec_model_setup(frame)
        
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Apply a face detection algorithm to get the face region
        # faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # self.utils.face_time(faces, frame)

        if len(detections) == 0:  # early return if detections is None or empty
            print("No detections")
            return

        final_canvas = self.utils.create_canvas(height, width)
        valid_detection = False

        for f in range(detections.shape[2]):
            confidence = detections[0, 0, f, 2]
            if confidence <= 0.5:
                
                continue

            box = detections[0, 0, f, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            if startX >= endX or startY >= endY:
                continue  # skip to the next iteration

            human_body = frame[startY:endY, startX:endX]
            if human_body.size <= 0:
                continue
            
            bw_human_body = cv2.cvtColor(human_body, cv2.COLOR_BGR2GRAY)
            self.past_human_detection = bw_human_body
            valid_detection = True  # valid detection
            # self.past_frame = frame

            new_height, new_width = bw_human_body.shape
            white_canvas = self.utils.create_canvas(new_height, new_width)

            self.curr_f = f
            col_change = self.utils.body_moved(detections, height, width, self.prev_box, self.prev_detection, self.curr_f)

            if col_change:
                self.utils.initialize_colors()

            for s in self.utils.get_steps_dynamic():
                self.draw_lines(bw_human_body, white_canvas, s)
            
            if white_canvas.shape != final_canvas[startY:endY, startX:endX].shape:
                print(f"Shape mismatch: white_canvas: {white_canvas.shape}, final_canvas slice: {final_canvas[startY:endY, startX:endX].shape}")
                continue
            
            final_canvas[startY:endY, startX:endX] = white_canvas
            self.col_clear_check = False
            self.utils.cv2_large(final_canvas, self.utils.screen_width, self.utils.screen_height)

            self.prev_box = box
            self.prev_detection = detections

        if not valid_detection:
            # self.past_frame = None
            self.utils.cv2_large(frame, self.utils.screen_width, self.utils.screen_height)

    

    def process_frame(self, frame):
        self.draw_body_detections(frame)

    
    def run(self):
        # if self.past_frame:          
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #     # Apply a face detection algorithm to get the face region
        #     faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        #     self.utils.face_time(faces, frame)
        while True:
            if self.past_frame is not None:  
                gray = cv2.cvtColor(self.past_frame, cv2.COLOR_BGR2GRAY)

        #     # Apply a face detection algorithm to get the face region
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)        
                self.utils.face_time(self.past_frame, faces)
                if self.utils.res_change:
                    self.past_frame = None

            ret, frame = self.utils.cap.read()
            if not ret:
                break
            self.past_frame = frame
            self.process_frame(frame)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                self.utils.cap.release()
                cv2.destroyAllWindows()
                break
        self.release_resources()

def actual_run():
    runner = DynamicRunner()
    runner.run()

if __name__ == "__main__":
    actual_run()
