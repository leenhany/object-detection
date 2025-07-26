
import cv2
import mediapipe as mp
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


##STREAMLIT SETUP
st.set_page_config(page_title="Object Detection App", page_icon=":guardsman:", layout="wide")
st.title("Object Detection App")
st.write("Upload an image to detect objects using EfficientDet Lite0 model.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","wbmp","jfif","webp"])
# STEP 1: main function 
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img=cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     st.image(img, channels="RGB", caption="Uploaded Image")

#     # STEP 3: load the image and run the detector.
#     mp_img=mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
#     detection_result = detector.detect(mp_img)

#     # STEP 4: detect objects and visualize the results.
#     annotated_image = visualize(mp_img, detection_result)
#     st.image(annotated_image, channels="RGB", caption="Annotated Image")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption='Uploaded Image')

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(mp_img)
    annotated_image = visualize(img, detection_result)
    st.image(annotated_image, caption='Annotated Image')
