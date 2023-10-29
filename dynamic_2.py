#Does not switch to face detection if face is too small

import cv2
from cv2 import CascadeClassifier, COLOR_BGR2GRAY, line, imshow, waitKey, destroyAllWindows  # More specific imports
import numpy as np
from utils import Utils

class DynamicRunner:
    def __init__(self):
        self.utils = Utils()
        self.col_clear_check = True
        self.f_dict = {}
        
        
    def release_resources(self):
        self.utils.cap.release()
        cv2.destroyAllWindows()


    def draw_lines(self, bw_face_neck, white_canvas, s):
        step = s['step']
        color = s['color']
        lower, upper = s['range']
        half_step = step // 2

        mask = (bw_face_neck[::step, ::step] >= lower) & (bw_face_neck[::step, ::step] < upper)
        y, x = np.where(mask)
        y = y * step
        x = x * step

        num_lines = len(x)
        points = np.zeros((num_lines, 2, 2), dtype=np.int32)

        points[:, 0, 0] = x - half_step
        points[:, 0, 1] = y - half_step
        points[:, 1, 0] = x + half_step
        points[:, 1, 1] = y + half_step

        cv2.polylines(white_canvas, points, isClosed=False, color=color, thickness=1)


    def draw_body_detections(self, frame):
        detections, height, width = self.utils.detec_model_setup(frame)
        steps = self.utils.steps

        if len(detections) == 0:  # early return if detections is None or empty
            print("No detections")
            return

        final_canvas = self.utils.create_canvas(height, width)
        valid_detection = False

        for f in range(detections.shape[2]):
            
            confidence = detections[0, 0, f, 2]
            if confidence <= 0.6:
                continue

            box = detections[0, 0, f, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            if startX >= endX or startY >= endY:
                continue  # skip to the next iteration

            human_body = frame[startY:endY, startX:endX]
            if human_body.size <= 0:
                continue
            
            bw_human_body = cv2.cvtColor(human_body, cv2.COLOR_BGR2GRAY)
            valid_detection = True  # valid detection

            new_height, new_width = bw_human_body.shape
            white_canvas = self.utils.create_canvas(new_height, new_width)
            
            f_exists = f in self.f_dict
            if f_exists:
                col_change = self.utils.body_moved(box, *self.f_dict[f][:2])

                if col_change:
                    self.utils.initialize_colors()
                    steps = self.utils.get_steps()
                    self.f_dict[f] = [detections, box, steps]

                for s in self.f_dict[f][2]:
                    self.draw_lines(bw_human_body, white_canvas, s)
            else:
                steps = self.utils.get_steps()
                self.f_dict[f] = [detections, box, steps]
                for s in self.f_dict[f][2]:
                    self.draw_lines(bw_human_body, white_canvas, s)
 
            if white_canvas.shape != final_canvas[startY:endY, startX:endX].shape:
                print(f"Shape mismatch: white_canvas: {white_canvas.shape}, final_canvas slice: {final_canvas[startY:endY, startX:endX].shape}")
                continue
            
            final_canvas[startY:endY, startX:endX] = white_canvas
            self.col_clear_check = False
            self.utils.cv2_large(final_canvas, self.utils.screen_width, self.utils.screen_height)

        if not valid_detection:
            self.utils.cv2_large(frame, self.utils.screen_width, self.utils.screen_height)
            # self.f_dict = {}
            self.col_clear_check = True
            if self.col_clear_check:
                self.utils.initialize_colors()

    

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
