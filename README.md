# laughing_man

convert images, video, or camera feed with faces into the laughing man

## Instructions

1. Place images you want to process inside the `app/process_folder` folder.

  - If you want to change the overlay image to something other than laughing man, replace the `/app/overlays/laughing_man_still.png` for images or the `/app/overlays/laughing_man.gif` for video/webcam.

2. Build the container:
`docker build -t laughingman .`

3. Run the container:
`docker run -v <path>/laughing_man/app:/app -it laughingman`

4. Processed images will be in the `app/output` folder.

Example input:

![Input](https://github.com/kudosudo/laughing_man/blob/main/app/process_folder/example.jpg "Example Input")

Example Output:

![Output](https://github.com/kudosudo/laughing_man/blob/main/app/output/example_image_0.png "Example Output")

Next Steps:

1. Add camera feed integration.

2. Add tdqm progress bar when processing video. Can take a bit and I wonder if it stalled when no output.

3. Simplify which mode to process for the end user.

4. Fix the overlay disappearing when bounding box is outside of background image dimensions.

5. GPU support to speed up processing time. Is slow for large images with high frame rates.
