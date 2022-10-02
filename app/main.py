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
  

def add_overlay(img:np.ndarray, bbox:list, overlay=None)->np.ndarray:
  # Get the height and width of the original image
  h, w, _ = img.shape
  # Open the image to overlay onto the bg
  if overlay is None:
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
  IMAGE_FILES = get_files_to_process(fpath, mode='image')
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

def gif_to_png():
  print('Processing gif into pngs.')
  # Where the gif is located
  path = './app/overlays/laughing_man.gif'
  # Where we should store our gif frames
  save_path = './app/overlays/gif_pngs/'
  # Read the gif
  gif_reader = imageio.get_reader(path)
  # Find out how many frames are in the gif
  gif_length = gif_reader.get_length()
  # print(f'total number of frames in the gif : {gif_length}')
  # Init frame count
  frame_index = 0
  while True:
      if frame_index < gif_length:
          # Load the gif frame
          frame = gif_reader.get_data(frame_index)
          # prep save path for frame_index 
          save_name = f'{save_path}frame_{frame_index}.png'
          # print(f'frame {frame_index}({frame.shape}): {save_name}')
          # Save the image and keep it transparent
          imageio.imwrite(save_name, frame, format='PNG-PIL')
          # Increase frame count
          frame_index = frame_index + 1
      else:
        break
  print('GIF successfully turned into pngs')
  return gif_length

def get_frames_per_second(path):
  cap = cv2.VideoCapture(path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  return fps

def get_frame_rate_diff(vid_fps:float, gif_fps:float)->int:
  rate_diff = vid_fps//gif_fps
  return int(rate_diff)
          
    

def process_video(gif_length:int)-> float:
  # Initialize mediapipe face detection
  mp_face_detection = mp.solutions.face_detection
  # Get list of files to process: 
  fpath = './app/process_folder/'
  VID_FILES = get_files_to_process(fpath, mode='video')
  VID_FILES = [file for file in VID_FILES if str(file).endswith('.mp4')]
  print(f'Total vids to process: {len(VID_FILES)}')
  print(VID_FILES)
  assert len(VID_FILES)==1, 'Please process one video at a time'
  vid = VID_FILES[0]
  # Load in the gif frames
  gif_frames = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in glob.glob("./app/overlays/gif_pngs/*.png")]
  print(f'Num of gif_frames: {len(gif_frames)}')
  with mp_face_detection.FaceDetection(
      model_selection=1, 
      min_detection_confidence=0.5
      ) as face_detection:
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(fpath+vid)
    # Get the Frames per second for both vid and gif
    fps = cap.get(cv2.CAP_PROP_FPS)
    gif_fps = get_frames_per_second('./app/overlays/laughing_man.gif')
    # Get the rate difference to speed up/slow down the gif relative to the video
    fps_rate_diff = get_frame_rate_diff(fps, gif_fps)
    print(f"vid_fps: {fps}, gif_fps: {gif_fps} -> {fps_rate_diff}")
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    # Start indices to track the frames for video and gif
    idx = 0
    gif_idx = 0
    # Increase this counter until it = frame_rate_diff.
    gif_rate_adjust_idx = 1
    # Read until video is completed
    print('Processing video. Please wait...')
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
              overlay_frame = gif_frames[gif_idx]
              # print(f'OVERLAY_FRAME: {overlay_frame.shape}')
              overlay_image = add_overlay(frame, bb_box, overlay=overlay_frame)
              # Increase the gif rate adjustment and gif frame counters
              if gif_rate_adjust_idx >= fps_rate_diff:
                gif_rate_adjust_idx = 1
                if gif_idx >= gif_length-1:
                  gif_idx = 0
                else:
                  gif_idx +=1
              else:
                gif_rate_adjust_idx+=1
            # Save the overlay frame
            cv2.imwrite('./app/tmp/frame_' + str(idx) + '.png', overlay_image)
            # Increase video frame idx counter
            idx+=1
          except Exception as e: 
            print(e)
            print(f'Failed to process frame {idx}')
            cv2.imwrite('./app/tmp/frame_' + str(idx) + '.png', frame)
            idx+=1
          if idx>=30:
            continue
      else:
        break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    print('Video processing complete.')
  return fps 
        
def clean_tmp_images(path:str):
  file_list = [file for file in os.listdir(path) if file!='.gitkeep']
  for file in file_list:
    os.remove(path+file)        

def convert_images_to_video(fps):
  print('Converting processed frames into video')
  path = './app/tmp/'
  file_list = [file for file in os.listdir(path) if str(file).startswith('frame_')]
  file_list = natsorted(file_list)
  writer = imageio.get_writer('./app/output/test_video.mp4', fps=fps)
  for file in file_list:
      im = imageio.imread(path+file)
      writer.append_data(im)
  writer.close()
  print('Recreated video with overlay complete.')
  print('Cleaning up')
  clean_tmp_images(path)
  print('Cleaning Complete')

def main(mode:str):
  print(f'Starting to process: {mode}')
  if mode == 'image':
    process_images()
  elif mode == 'video':
    gif_length = gif_to_png()
    fps = process_video(gif_length=gif_length)
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
  main(mode=mode)