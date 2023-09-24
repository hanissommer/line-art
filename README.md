# line-art
 **Detecting faces (and bodies) and drawing them with lines**

 This is a series of programs bringing to fruition an art idea of mine -- recreating the human body/face with a meticulous orchestration of straight lines.

 The idea grew into the motivation of capturing and recreating the contour of our bodies through processing the different pixel regions (shades) that are cast on us as we interact with light. With precise calculations to deduce the length, color, and location of numerous straight lines, the human body becomes art.

 The first file, 'lineArt', started it all where the user will be prompted to select a locally stored image that the program will process. The face is identified, if any is present in the image, and based on the pixel coverage of the grayscale version of the image, the face will be reconstructed using 45-degree angled black lines, atop a white canvas. This canvas with the reconstructed face is then given as an output.

 The second file, 'lineArt1', performs similarly to the prior, only that it stores the location of the face from the input image and pastes the final canvas back onto the original image at the same position. This results in the output image having the background (with color) intact while the detected face will be replaced by the line-filled replica, drawn on a white canvas.

 The third file, 'lineArtLive_blackWhite', uses a live feed from a webcam instead of requesting an image. For each frame of the video, similar processing as in the program prior is done to produce an output frame with a line-filled face. This processing is done in real-time.

 The fourth file, 'lineArtLiveChanging_ColorFrequent', is similar to the one prior with the use of a live feed from a webcam and processing each frame of the video to produce an output frame with a line-filled face. However, the color of the lines used is not black but is randomly (I don't remember the logic) selected from a given color palette. A new color is chosen for each frame processing, resulting in each frame having a different array of colors.

 The fifth file, 'lineArtLiveChanging_Color', is similar to the one prior but, the color of the lines used only changes when there is a new face detection.

 In the sixth file, 'youAreArt_colorFrequent_Large', instead of just a face, a human body is used to create a line drawing. When a body isn't detected, the normal camera frame is displaced, but when a body becomes detected, the background is striped away and only the line art of the body/bodies is shown. Similar to 'lineArtLiveChanging_ColorFrequent', the color of the lines used changes every frame. Lastly, the output frame of this program is set to be enlarged as much as the display allows which may affect resolution.  

 In the file, 'youAreArt_colorFrequent_small', the same logic of the program prior applies but, the output frame has a defined size which is small but has great resolution.

 In 'youAreAre_colorLarge', the same logic as 'youAreArt_colorFrequent_Large' applies but the colors used for the lines are only changed whenever the 'body detected' status changes.

 In 'youAreAre_colorSmall', the same logic as 'youAreArt_colorFrequent_small' applies but the colors used for the lines are only changed whenever the 'body detected' status changes.


  

