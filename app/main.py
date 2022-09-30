import cv2
import mediapipe as mp
import math
from typing import List, Mapping, Optional, Tuple, Union

def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def scale_bounding_box(bbox):
  box_x, box_y, box_w, box_h = bbox
  x0 = box_x
  x1 = box_x+box_w
  y0 = box_y
  y1 = box_y+box_h
  # scale the box size
  scale_w = box_w * 1.5
  scale_h = box_h * 1.5
  # Get centre of original shape, and position of top-left of ROI in output image
  cx, cy = (x0 + x1) /2, (y0 + y1)/2
  top  = cy - scale_h/1.8
  left = cx - scale_w/2
  return left, top, scale_w, scale_h
  

def add_overlay(img, bbox):
  # Get the height and width of the original image
  h, w, _ = img.shape
  # Open the image to overlay onto the bg
  overlay = cv2.imread('app/imgs/laugh_man_still.png', cv2.IMREAD_UNCHANGED)
  # Get the sizes and location of the bounding box
  # box_x, box_y, box_w, box_h = bbox
  box_x, box_y, box_w, box_h = scale_bounding_box(bbox)
  # Convert box_w & box_y to pixel vals
  box_w, box_h = normalized_to_pixel_coordinates(box_w, box_h, w, h)
  # print(f'box size: ({box_w},{box_h})')
  # Get pixel coordinates of the bounding box
  box_x, box_y = normalized_to_pixel_coordinates(box_x, box_y, w, h)
  # print(f'bbox coords: ({box_x},{box_y})')
  # resize the overlay to the bounding box dims
  overlay = cv2.resize(overlay, (box_w, box_h))
  # print(f'Size of overlay: {overlay.shape}')
  # print(f'Size of image: {img.shape}')
  result = img.copy()
  # Making the overlay background transparent
  alpha_s = overlay[:, :, 3] / 255.0
  alpha_l = 1.0 - alpha_s
  # Add the overlay on the coordinates provided
  for c in range(0, 3):
    result[box_y:box_y+box_h, box_x:box_x+box_w, c] = (alpha_s * overlay[:, :, c] +
                              alpha_l * result[box_y:box_y+box_h, box_x:box_x+box_w, c])
  return result
  
  

def main():
  # Initialize mediapipe face detection and drawing onto screen
  mp_face_detection = mp.solutions.face_detection
  
  # Images to process. 
  IMAGE_FILES = ['app/imgs/example.jpg']
  with mp_face_detection.FaceDetection(
      model_selection=1, 
      min_detection_confidence=0.5
      ) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
      # Read the file
      image = cv2.imread(file)
      # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      # Draw face detections of each face.
      if not results.detections:
        continue
      # # Create a copy of the image
      # annotated_image = image.copy()
      for detection in results.detections:
        # Get the bounding box dimensions used
        location_data = detection.location_data
        # print(f'location_data: {location_data}')
        if location_data.format == location_data.RELATIVE_BOUNDING_BOX:
            bb = location_data.relative_bounding_box
            bb_box = [
                bb.xmin, bb.ymin,
                bb.width, bb.height,
            ]
            print(f"RBBox: {bb_box}")
        else:
          print('no box detected')
        # Instead of drawing, use the space below to add the overlay
        overlay_image = add_overlay(image, bb_box)
      cv2.imwrite('./app/output/example_image_' + str(idx) + '.png', overlay_image)
  
  
  
if __name__ == "__main__":
  main()

