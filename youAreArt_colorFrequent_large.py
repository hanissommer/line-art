import cv2
import numpy as np
from utils import Utils

def draw_lines(bw_face_neck, white_canvas, s):
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

def run_yarcfl():
    utils = Utils()

    while True:
        ret, frame = utils.cap.read()
        if not ret:
            break

        detections, height, width = utils.detec_model_setup(frame)
        utils.initialize_colors()
        if len(detections) == 0:  # early return if detections is None or empty
            print("No detections")
            return

        final_canvas = utils.create_canvas(height, width)
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
            white_canvas = utils.create_canvas(new_height, new_width)

            for s in utils.get_steps_dynamic():
                draw_lines(bw_human_body, white_canvas, s)
            
            if white_canvas.shape != final_canvas[startY:endY, startX:endX].shape:
                print(f"Shape mismatch: white_canvas: {white_canvas.shape}, final_canvas slice: {final_canvas[startY:endY, startX:endX].shape}")
                continue
            
            final_canvas[startY:endY, startX:endX] = white_canvas

            utils.cv2_large(final_canvas, utils.screen_width, utils.screen_height)


        if not valid_detection:
            utils.cv2_large(frame, utils.screen_width, utils.screen_height)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            utils.cap.release()
            cv2.destroyAllWindows()
            break

    utils.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yarcfl()