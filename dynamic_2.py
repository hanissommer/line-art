import cv2
from cv2 import CascadeClassifier, COLOR_BGR2GRAY, line, imshow, waitKey, destroyAllWindows  # More specific imports
import numpy as np
from utils import Utils

class DynamicRunner:
    def __init__(self):
        self.utils = Utils()
        self.screen_width, self.screen_height = self.utils.get_monitor_details(1)
        # self.cap = cv2.VideoCapture(1)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

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
    

    def draw_body_detections(self, frame):
        detections, height, width = self.utils.detec_model_setup(frame)
        final_canvas = self.utils.create_canvas(height, width)

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

                            col_change = self.utils.body_moved(detections, height, width, self.prev_box, self.prev_detection)

                            if col_change is True:
                                self.utils.initialize_colors()

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
                        
                            self.prev_box = box
                            self.prev_detection = detections
                        
                        else:
                            self.utils.cv2_large(frame, self.screen_width, self.screen_height)


            if valid_detection:
                self.col_clear_check = False
                # self.prev_detection = detections
                self.utils.cv2_large(final_canvas, self.screen_width, self.screen_height)
                
            else:
                self.utils.cv2_large(frame, self.screen_width, self.screen_height)  # If no valid detection, display the original frame
                if self.col_clear_check == False:
                    self.utils.clear_colors()
                    self.col_clear_check = True

    

    def process_frame(self, frame):
        self.draw_body_detections(frame)

    
    def run(self):

        while True:
            ret, frame = self.utils.cap.read()
            if not ret:
                break
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
