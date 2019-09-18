import cv2
import numpy as np

# Load the Haar Cascad files for face and eye
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Check if the face cascade file has been loaded correctly
if face_cascade.empty():
    raise IOError('Unable to load the face cascade file')

# Check if the eye cascade file has been loaded correctly
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade file')

# Initialise the video capture object
cap = cv2.VideoCapture('head-pose-face-detection-male.mp4') # Put 0 as arg for video cam

# Define the scaling vector
ds_factor = 0.5

# Iterate until the user hits escape key:
while True:
    # Capture the current frame
    _, frame = cap.read()
    if(_):
        # Resize the frame
        frame = cv2.resize(frame, None, fx = ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run the face detector on the grayscale img
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # For each face that's detected, run the eye detector
        for (x,y,w,h) in faces:
            
            # Extract the grayscale face ROI
            roi_gray = gray[y:y+h, x:x+h]
            
            # Extract the color face ROI
            roi_color = frame[y:y+h, x:x+h]
            
            # Run eye detector on grayscale ROI
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # Draw circles around the eye
            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                radius = int(0.3*(w_eye+h_eye))
                color = (0,255,0)
                thickness = 3
                cv2.circle(roi_color, center, radius, color, thickness)
                
        # Display the output
        cv2.imshow('Eye Detector', frame)
        
        # Check if the iser hit the escape key
        c = cv2.waitKey(1)
        if c==27:
            break
    else:
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
    