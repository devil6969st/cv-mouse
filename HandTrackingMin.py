import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS
      )

    # Get the top left corner of the detected hand's bounding box.
    # height, width, _ = annotated_image.shape
    # x_coordinates = [landmark.x for landmark in hand_landmarks]
    # y_coordinates = [landmark.y for landmark in hand_landmarks]
    # text_x = int(min(x_coordinates) * width)
    # text_y = int(min(y_coordinates) * height) - MARGIN
    #
    # # Draw handedness (left or right hand) on the image.
    # cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

base_options = python.BaseOptions(model_asset_path=r"S:\Downloads\hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options,num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0
while True:
    _,img = cap.read()
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    detection_result = detector.detect(mp_img)
    annotated_image = draw_landmarks_on_image(img_rgb, detection_result)
    # cv2.imshow("Image",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(annotated_image,str(int(fps)), (70,70), cv2.FONT_HERSHEY_DUPLEX,3,(255,0,255),2)
    cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.destroyAllWindows()


