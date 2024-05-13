import cv2
import mediapipe as mp
import math
from collections import deque
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import random

def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

def update_landmark_position(prev_landmark, curr_landmark, threshold=0.03):
    # Update landmark position if it has moved more than the threshold distance
    distance = calculate_distance(prev_landmark, curr_landmark)
    if distance > threshold:
        return curr_landmark
    else:
        return prev_landmark
def zoom_in(driver, zoom_percentage):
    driver.execute_script(f"document.body.style.zoom = {-101}")
    print("zoom in"+"document.body.style.zoom = {0}%".format(100 + zoom_percentage))
    
def zoom_out(driver, zoom_percentage):
    driver.execute_script(f"document.body.style.zoom = {-zoom_percentage}")
    print("document.body.style.zoom = {0}%".format(100 - 50 * zoom_percentage))

def scroll_up(driver, speed,  distance):
    distance = (random.random() + speed)*10 + distance
    driver.execute_script("window.scrollBy(0, {0})".format(-distance))
    print("window.scrollBy(0, {0})".format(-distance))

def scroll_down(driver, speed, distance):
    distance = (random.random() + speed)*10 + distance
    driver.execute_script("window.scrollBy(0, {0})".format(distance))
    print("window.scrollBy(0, {0})".format(distance))

def set_driver():
    # Set up the Chrome driver
    driver = webdriver.Chrome()
    driver.get("https://en.wikipedia.org/wiki/Adolf_Hitler")
    return driver

def main():
    driver = set_driver()
    # Initialize MediaPipe Hand model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)
    
    # Initialize MediaPipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize OpenCV VideoCapture
    cap = cv2.VideoCapture(0)
    
    
    prev_landmarks = None
    prev_distances = deque(maxlen=15)  # Store distances from last 2 frames
    frame_count = 0
    prev_time = time.time()

    while cap.isOpened():
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        speed = None
        action = "No Action"
        lock = False
        count = 1
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Update landmark positions only if they have moved more than 10 pixels
                if count % 10 == 0 : 
                    lock = True
                if prev_landmarks is not None:
                    for landmark, prev_landmark in zip(hand_landmarks.landmark, prev_landmarks.landmark):
                        landmark.x, landmark.y = update_landmark_position(prev_landmark, landmark).x, update_landmark_position(prev_landmark, landmark).y
                
                # Store current landmarks for the next iteration
                prev_landmarks = hand_landmarks
                
                # Calculate distance between index finger tip and thumb tip
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle  =  hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                

                if calculate_distance(index_tip, middle) < 0.05:
                    distance = calculate_distance(index_tip, wrist)
                else:
                    distance = calculate_distance(index_tip, thumb_tip)
                # speed 
                if len(prev_distances) > 0:
                    prev_distance = prev_distances[-1]
                    curr_time = time.time()
                    frame_count += 1
                    if frame_count > 1:
                        time_diff = curr_time - prev_time
                        speed = 10 * abs(distance - prev_distance) / time_diff
                    
                
                # Compare distance with distance from k frames ago
                k = 5
                if speed is not None and speed < 0.02:
                    if calculate_distance(index_tip, middle) < 0.08:
                        distance = calculate_distance(index_tip, wrist)
                        if len(prev_distances) > k+1:
                            prev_distance = prev_distances[-k]
                            if distance > prev_distance:
                                action = "scroll down"
                            elif distance < prev_distance:
                                action = "scroll up"
                            else:
                                action = "No Action"
                    else:
                        distance = calculate_distance(index_tip, thumb_tip)
                        if len(prev_distances) > k+1:
                            prev_distance = prev_distances[-k]
                            if distance > prev_distance:
                                action = "zoom in"
                            elif distance < prev_distance:
                                action = "zoom out"
                            else:
                                action = "No Action"
                else:
                    action = "No Action"
                    
                # Update previous distances
                prev_distances.append(distance)
                count+=1
        #driver = None 
        # Draw landmarks on the frame
        if speed is not None:
            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if action is not "No Action":
                    if action == "scroll up":
                        scroll_up(driver,speed,  200)
                    elif action == "scroll down":
                        scroll_down(driver,speed,  100)
                    elif action == "zoom in":
                        zoom_in(driver, 1)
                    elif action == "zoom out":
                        zoom_out(driver, 1)
                # Display action label on the frame
                cv2.putText(frame, action, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Speed: {speed:.2f} pixels/frame", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        time.sleep(0.1)
        # Display the frame
        cv2.imshow('Hand Tracking', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
