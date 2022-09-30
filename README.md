# laughing_man
convert images, video, or camera feed with faces into the laughing man

To build the container:
`docker build -t laughingman .`

To run the container:
`docker run -v <path>/laughing_man/app:/app -it laughingman`

Example input: 
![Input](https://github.com/kudosudo/laughing_man/blob/main/app/imgs/example.jpg "Example Input")
![Output](https://github.com/kudosudo/laughing_man/blob/main/app/output/example_image_0.png "output")