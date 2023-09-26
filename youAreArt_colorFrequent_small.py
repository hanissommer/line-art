import cv2
import numpy as np
from utils import detec_model_setup, clear_colors, create_canvas, initialize_colors, get_steps

def run_yarcfs():
    col_clear_check = True
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, height, width = detec_model_setup(frame)
        initialize_colors()
        final_canvas = create_canvas(height, width)

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

                            height, width = bw_face_neck.shape
                            white_canvas = create_canvas(height, width)

                            # Drawing lines based on pixel color conditions
                            for s in get_steps():
                                step = s['step']
                                color = s['color']
                                lower, upper = s['range']
                                
                                # Create a mask where the conditions are met
                                mask = (bw_face_neck[::step, ::step] >= lower) & (bw_face_neck[::step, ::step] < upper)
                                
                                # Get the indices where mask is True and adjust the coordinates
                                y, x = np.where(mask)
                                y = y * step
                                x = x * step
                                
                                # Draw lines on the white_canvas
                                for i, j in zip(x, y):
                                    cv2.line(white_canvas, (i - step//2, j - step//2), (i + step//2, j + step//2), color, 1)

                            # After processing all steps, assign white_canvas to the appropriate location on final_canvas
                            if white_canvas.shape == final_canvas[startY:endY, startX:endX].shape:
                                final_canvas[startY:endY, startX:endX] = white_canvas
                            else:
                                cv2.imshow('yarcfs', frame)
                                print(f"Shape mismatch: white_canvas: {white_canvas.shape}, final_canvas slice: {final_canvas[startY:endY, startX:endX].shape}")
                        else:
                            cv2.imshow('yarcfs', frame)

            if valid_detection:
                col_clear_check = False
                cv2.imshow('yarcfs', final_canvas)
            else:
                cv2.imshow('yarcfs', frame)  # If no valid detection, display the original frame
                if col_clear_check == False:
                    clear_colors()
                    col_clear_check = True

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_yarcfs()