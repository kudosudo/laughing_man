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
