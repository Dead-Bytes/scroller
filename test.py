import cv2
import mediapipe as mp
import math
import time

def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

def zoom_in():
    print("Zoom in function called")
def zoom_out():
    print("Zoom out function called")
def update_landmark_position(prev_landmark, curr_landmark, threshold=10):
    # Update landmark position if it has moved more than the threshold distance
    distance = calculate_distance(prev_landmark, curr_landmark)
    if distance > threshold:
        return curr_landmark
    else:
        return prev_landmark

def main():
    # Initialize MediaPipe Hand model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Initialize OpenCV VideoCapture
    cap = cv2.VideoCapture(0)
    
    prev_distance = None
    prev_time = time.time()
    frame_count = 0
    
    while cap.isOpened():
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        pinching = False
        speed = None
        scrolling = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the index finger and thumb
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]  
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Calculate the distance between the index finger tip and thumb tip
                distance = calculate_distance(index_tip, thumb_tip)
                distance_1 = calculate_distance(index_tip, middle_tip)
                distance_2 = calculate_distance(middle_tip, wrist)
                distance_3 = calculate_distance(index_tip, wrist)
                
                # Check if the distance is increasing compared to the previous frame
                # if prev_distance is not None:
                #     if distance > prev_distance:
                #         pinching = False  # Fingers moving apart
                #     else:
                #         pinching = True   # Fingers moving closer or staying the same
                if prev_distance is not None:
                    if distance_3 > prev_distance  :
                        scrolling = "scrolling up"
                    if distance_3 < prev_distance  :
                        scrolling = "scrolling down"
                # Calculate relative speed
                curr_time = time.time()
                frame_count += 1
                if frame_count > 1:
                    time_diff = curr_time - prev_time
                    speed = abs(distance_3 - prev_distance) / time_diff
                
                prev_distance = distance_3
                prev_time = curr_time
        
        # Display pinching status and speed on the frame
        if speed is not None:
            cv2.putText(frame, f"Speed: {speed:.2f} pixels/frame", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if speed is not None and speed < 0.5: 
            # if pinching:
            #     cv2.putText(frame, "Moving closer", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #     cv2.putText(frame, "Zoom in function to call", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     zoom_in()
            # else:
            #     cv2.putText(frame, "Moving apart", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     cv2.putText(frame, "Zoom out function to call", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     zoom_out()
            if scrolling is not None:
                cv2.putText(frame, scrolling, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:   
            cv2.putText(frame, "Speed too fast", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
                
        # Display the frame
        cv2.imshow('Hand Tracking', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
