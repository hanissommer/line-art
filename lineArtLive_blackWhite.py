import cv2
import numpy as np
from utils import Utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_lines(bw_face_neck, white_canvas, s):
        step = s['step']
        color = 0
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

def run_lalbw():
    utils = Utils()

    while True:
        #Get the frame
        ret, frame = utils.cap.read()
        if not ret:
            break
        height, width, channels = frame.shape
        final_canvas = utils.create_canvas(height, width)
        final_canvas[:height, :width] = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        if len(faces) == 0:
            utils.cv2_large(frame, utils.screen_width, utils.screen_height)
        
        for x, y, w, h in faces:
            bw_face_neck = gray[y:y+h, x:x+w]  # reusing the already converted gray image
            white_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
              
            for s in utils.get_steps():
                draw_lines(bw_face_neck, white_canvas, s)
                
            final_canvas[y:y+h, x:x+w] = white_canvas
        
        utils.cv2_large(final_canvas, utils.screen_width, utils.screen_height)


        # Check for key events
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' or ESC key
            utils.cap.release()
            cv2.destroyAllWindows()
            break
    utils.cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_lalbw()