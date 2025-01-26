import cv2
import numpy as np
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Create a canvas for drawing
canvas = None
brush_size = 10
color = (0, 0, 255)  # Default color (Red)
eraser_mode = False

# Function to calculate distance between two fingers
def finger_distance(finger1, finger2):
    return np.linalg.norm(np.array((finger1.x, finger1.y)) - np.array((finger2.x, finger2.y)))

# Function to draw the toolbar
def draw_toolbar(window_width):
    toolbar_height = 50
    toolbar = np.ones((toolbar_height, window_width, 3), dtype=np.uint8) * 255  # White toolbar

    # Define colors and their positions with spacing
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]
    spacing = 20
    for i, c in enumerate(colors):
        cv2.rectangle(toolbar, (i * (50 + spacing), 0), ((i + 1) * 50 + i * spacing, toolbar_height), c, -1)

    # Draw eraser, clear all, and save options
    eraser_start = (len(colors) + 1) * (50 + spacing)  # Position for the eraser
    cv2.rectangle(toolbar, (eraser_start, 0), (eraser_start + 100, toolbar_height), (255, 255, 255), -1)  # Eraser
    cv2.putText(toolbar, "Eraser", (eraser_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    clear_start = eraser_start + 100 + spacing  # Position for clear all
    cv2.rectangle(toolbar, (clear_start, 0), (clear_start + 100, toolbar_height), (255, 255, 255), -1)  # Clear all
    cv2.putText(toolbar, "Clear All", (clear_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    save_start = clear_start + 100 + spacing  # Position for save
    cv2.rectangle(toolbar, (save_start, 0), (save_start + 100, toolbar_height), (255, 255, 255), -1)  # Save file
    cv2.putText(toolbar, "Save", (save_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return toolbar, eraser_start, clear_start, save_start

# Function to open save dialog
def save_dialog(canvas):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG files", "*.png"),
                                                          ("JPEG files", "*.jpg"),
                                                          ("All files", "*.*")])
    if file_path:
        cv2.imwrite(file_path, canvas)
        print(f"Canvas saved as {file_path}")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Set up full-screen window
cv2.namedWindow('AirCanvas', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('AirCanvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        height, width, _ = frame.shape
        
        if canvas is None:
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw the toolbar
        toolbar, eraser_start, clear_start, save_start = draw_toolbar(width)

        # Initialize finger position variables
        cx, cy = 0, 0  # Default values

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[8]
                middle_finger_tip = hand_landmarks.landmark[12]

                # Update cx and cy based on the position of the index finger tip
                cx, cy = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

                # Detect one-finger (draw) or two-finger (select) gestures
                if finger_distance(index_finger_tip, middle_finger_tip) > 0.05:
                    # One finger - Drawing mode
                    if not eraser_mode:
                        cv2.circle(canvas, (cx, cy), brush_size, color, cv2.FILLED)
                    else:
                        cv2.circle(canvas, (cx, cy), 30, (255, 255, 255), cv2.FILLED)
                else:
                    # Two fingers - Switch to selecting mode
                    for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]):
                        if i * (50 + 20) < cx < (i + 1) * 50 + i * 20 and 0 < cy < 50:
                            color = c
                            eraser_mode = False

                    if eraser_start < cx < eraser_start + 100 and 0 < cy < 50:
                        eraser_mode = True  # Eraser mode
                    elif clear_start < cx < clear_start + 100 and 0 < cy < 50:
                        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # Clear all
                    elif save_start < cx < save_start + 100 and 0 < cy < 50:
                        # Open save dialog
                        save_dialog(canvas)

                # Draw landmarks on the camera feed
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Ensure both frame and canvas are of the same type
        canvas = canvas.astype(frame.dtype)

        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Stack the toolbar and the combined image
        output = np.vstack((toolbar, combined))
        cv2.imshow('AirCanvas', output)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
