import cv2
import numpy as np
import timeit

#GPT4's second fix
def draw_lines(self, bw_face_neck, white_canvas, s):
   start_time = timeit.default_timer()

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

   elapsed_time = timeit.default_timer() - start_time
   print(f"Time taken: {elapsed_time} seconds")

#GPT4's initial fix
def draw_lines(self, bw_face_neck, white_canvas, s):
   start_time = timeit.default_timer()
   step = s['step']
   color = s['color']
   lower, upper = s['range']
   half_step = step // 2  # Calculate once outside the loop

   # Using slicing and vectorization
   mask = (bw_face_neck[::step, ::step] >= lower) & (bw_face_neck[::step, ::step] < upper)

   y, x = np.where(mask)
   y = y * step
   x = x * step
   # Assuming line() can accept arrays; if not, further optimization is needed.
   line(white_canvas, (x - half_step, y - half_step), (x + half_step, y + half_step), color, 1)

   elapsed_time = timeit.default_timer() - start_time
   print(f"Time taken: {elapsed_time} seconds")


#GPT4 addition to casey's half fix
def draw_lines(self, bw_face_neck, white_canvas, s):
   start_time = timeit.default_timer()
   step = s['step']
   color = s['color']
   lower, upper = s['range']

   mask = np.zeros_like(bw_face_neck, dtype=bool)
   mask[::step, ::step] = (bw_face_neck[::step, ::step] >= lower) & (bw_face_neck[::step, ::step] < upper)

   y, x = np.where(mask)
   for i, j in zip(x * step, y * step):  # Multiply by step to get back to the original coordinates
       half_step = step // 2  # calculate once outside the loop
       line(white_canvas, (i - half_step, j - half_step), (i + half_step, j + half_step), color, 1)

   elapsed_time = timeit.default_timer() - start_time
   print(f"Time taken: {elapsed_time} seconds")

#Current
def draw_lines(self, bw_face_neck, white_canvas, s):
        start_time = timeit.default_timer()
        step = s['step']
        color = s['color']
        lower, upper = s['range']
        mask = (bw_face_neck[::step, ::step] >= lower) & (bw_face_neck[::step, ::step] < upper)
        y, x = np.where(mask)
        y = y * step
        x = x * step
        for i, j in zip(x, y):
            line(white_canvas, (i - step // 2, j - step // 2), (i + step // 2, j + step // 2), color, 1)

        elapsed_time = timeit.default_timer() - start_time
        print(f"Time taken: {elapsed_time} seconds")
