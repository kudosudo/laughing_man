import os
import sys
import cv2
import mediapipe as mp
import math
import numpy as np
import imageio.v2 as imageio
import glob
from natsort import natsorted
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

def scale_bounding_box(bbox:list)-> float:
  # Get the location coords to place image and image width & height
  box_x, box_y, box_w, box_h = bbox
  # Get the upper-left and lower-right coordinates of the box
  x0 = box_x
  x1 = box_x+box_w
  y0 = box_y
  y1 = box_y+box_h
  # scale the box size
  scale_w = box_w * 1.5
  scale_h = box_h * 1.5
  # Get centre of original shape, and position of top-left of ROI in output image
  cx, cy = (x0 + x1) /2, (y0 + y1)/2
  top  = cy - scale_h/1.7
  left = cx - scale_w/2
  return left, top, scale_w, scale_h
  

def add_overlay(img:np.ndarray, bbox:list)->np.ndarray:
  # Get the height and width of the original image
  h, w, _ = img.shape
  # Open the image to overlay onto the bg
  overlay = cv2.imread('app/overlays/laugh_man_still.png', cv2.IMREAD_UNCHANGED)
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
  
def get_files_to_process(path:str, mode:str)->list:
  file_list = [file for file in os.listdir(path) if file!='.gitkeep']
  ## TO DO: 
  #  1. If video, convert into images stored in /app/tmp/
  #  2. Conf something for multiple video or mixed media
  return file_list



def process_images():
  # Initialize mediapipe face detection
  mp_face_detection = mp.solutions.face_detection
  
  # Get list of files to process: 
  fpath = './app/process_folder/'
  IMAGE_FILES = get_files_to_process(fpath, mode)
  IMAGE_FILES = [file for file in IMAGE_FILES if str(file).endswith('.jpg')]
  print(f'Total files to process: {len(IMAGE_FILES)}')
  print(IMAGE_FILES)
  with mp_face_detection.FaceDetection(
      model_selection=1, 
      min_detection_confidence=0.5
      ) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
      print(f'Processing: {file}')
      # Read the file
      image = cv2.imread(fpath+file)
      # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      # Ensure a face was detected
      if not results.detections:
        continue
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
            # print(f"RBBox: {bb_box}")
        else:
          print('no box detected')
        # Instead of drawing, use the space below to add the overlay
        overlay_image = add_overlay(image, bb_box)
      cv2.imwrite('./app/output/example_image_' + str(idx) + '.png', overlay_image)


def process_video():
  # Initialize mediapipe face detection
  mp_face_detection = mp.solutions.face_detection
  # Get list of files to process: 
  fpath = './app/process_folder/'
  VID_FILES = get_files_to_process(fpath, mode)
  VID_FILES = [file for file in VID_FILES if str(file).endswith('.mp4')]
  print(f'Total vids to process: {len(VID_FILES)}')
  print(VID_FILES)
  assert len(VID_FILES)==1, 'Please process one video at a time'
  vid = VID_FILES[0]
  with mp_face_detection.FaceDetection(
      model_selection=1, 
      min_detection_confidence=0.5
      ) as face_detection:
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(fpath+vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    # Read until video is completed
    idx = 0
    while(cap.isOpened()):
      # print(f'...processing frame {idx}')
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Ensure a face was detected
        if not results.detections:
          print(f'no face detected for frame {idx}')
          cv2.imwrite('./app/tmp/frame_' + str(idx) + '.png', frame)
          idx+=1
        else:
          try:
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
                  # print(f"RBBox: {bb_box}")
              else:
                print('no box detected')
              # Instead of drawing, use the space below to add the overlay
              overlay_image = add_overlay(frame, bb_box)
            cv2.imwrite('./app/tmp/frame_' + str(idx) + '.png', overlay_image)
            idx+=1
          except:
            print(f'Failed to process frame {idx}')
            cv2.imwrite('./app/tmp/frame_' + str(idx) + '.png', frame)
            idx+=1
          # if idx>=30:
          #   break
      else:
        break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
  return fps 
        
def clean_tmp_images(path:str):
  file_list = [file for file in os.listdir(path) if file!='.gitkeep']
  for file in file_list:
    os.remove(path+file)        

def convert_images_to_video(fps):
  path = './app/tmp/'
  file_startswith = 'frame_'
  file_list = [file for file in os.listdir(path) if str(file).startswith('frame_')]
  file_list = natsorted(file_list)
  writer = imageio.get_writer('./app/output/test_video.mp4', fps=fps)
  # for file in glob.glob(os.path.join(path,f'{file_startswith}*.png')):
  for file in file_list:
      im = imageio.imread(path+file)
      writer.append_data(im)
  writer.close()
  clean_tmp_images(path)

def main(mode:str):
  print(f'Starting to process: {mode}')
  if mode == 'image':
    process_images()
  elif mode == 'video':
    fps = process_video()
    convert_images_to_video(fps=fps)
  elif mode == 'cam':
    pass
  else:
    pass
  
  print('Processing Complete. Please find processed media in the "app/output/" folder')
  
  
  
if __name__ == "__main__":
  args = [arg for arg in  sys.argv if arg!='app/main.py']
  print(f'mode: {args}')
  assert len(args)==1, 'Please pass a single argument from this list: [image, video, cam]'
  mode = args[0]
  assert mode in ['image','video','cam'], 'mode must be one of: [image, video, cam]'
  main(mode='video')

