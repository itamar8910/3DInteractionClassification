# Problem definition
<b>Problem:</b>Given 4 videos from 4 different cameras, Detect type of interaction.
4 types of interaction
- No interaction
- Touch
- Look
- Both

<b>Challenge:</b> The system cannot assume anything about the scene <br/>
<p align="center"> <img src="https://image.ibb.co/e4AL9T/Screen_Shot_2018_07_24_at_11_25_30.png" width="50%"></p>

# Input
<p align="center"><img src="https://preview.ibb.co/eXVyw8/Screen_Shot_2018_07_24_at_11_42_42.png" width="50%"></p>
We receive as input 4 video cameras that are in the corner of the room. 

# First approach - 3D reconstruction
<p align="center"><img src="https://preview.ibb.co/cmwSpT/Screen_Shot_2018_07_24_at_11_47_07.png" width="75%"></p>
<br/>
We first calibrate our 4 cameras in-order to reconstructe the 3d scene.
<p align="center"><img src="https://preview.ibb.co/kfQh68/Screen_Shot_2018_07_24_at_15_29_01.png" width="75%"></p>

<b>Then, we perform the following steps:</b>
1. Detect person using openpose
2. Recognize person's identity
3. find (x,y) coordinates of both eyes and noise
4. Find (x,y,z) coordinates from two cameras
5. Find each person’s face plane
6. Get plane's normal => looking direction
7. Classify interaction
<p align="center"><img src="https://preview.ibb.co/cZUBYo/Screen_Shot_2018_07_24_at_15_36_46.png" width="100%"></p>

This approach worked well but had room for improvment. This is because of errors in the calibration. 
Because of this issues we improved to our second approach.
<p align="center"><img src="https://preview.ibb.co/n46VR8/Screen_Shot_2018_07_24_at_15_38_48.png" width="100%"></p>

# Second approach - 3D estimation
<p align="center"><img src="https://preview.ibb.co/jKXWYo/Screen_Shot_2018_07_24_at_15_46_24.png" width="100%"></p>
Instead of tring to reconstructe the 3D dimension, we will try to estimate them.
We use deep learning and Image proccssing in order to aproximate the data.

our steps:
1. Detect person using tinyFaces: using deep learning classification(tinyFace model) we can find the face dimensions of people in the scene with high accuracy even in very small and low level resolution.
<p align="center"><img src="https://preview.ibb.co/iwLwzT/Screen_Shot_2018_07_24_at_15_49_31.png" width="100%"></p>

2. Recognize person’s Identity: Using deep learning one shot person recognition and a fall back to HSV color detection we can identify the people in the scene(With a few pictures taken from them)
<p align="center"><img src="https://preview.ibb.co/itBato/Screen_Shot_2018_07_24_at_15_49_35.png" width="100%"></p>

3. Find distance of person from camera: Using the size of the person face and its body proportion extracted with openpose, we can estimate the user distance from the camera.
<p align="center"><img src="https://preview.ibb.co/nJPaR8/Screen_Shot_2018_07_24_at_15_49_39.png" width="100%"></p>

4. Get looking direction: Using gazer library which uses advanced deep learning and image proccessing techniques, we can find the gaze of the person. We also implemented a fall back heuristic based on the persons face to handle failure of gazer.
<p align="center"><img src="https://preview.ibb.co/d2Azm8/Screen_Shot_2018_07_24_at_15_49_43.png" width="100%"></p>

5. Classify interaction: finally, we can find the L2 distance between the two vectors from the peoples nose, and below a certain threshold alaram as interaction detection.
<p align="center"><img src="https://preview.ibb.co/dRUEKT/Screen_Shot_2018_07_24_at_15_56_02.png" width="100%"></p>

# Output demo
We are able to detect face interaction. 
Whether two people are looking at each other, who are the people and how confident are we about the classification.
<p align="center"><img src="https://image.ibb.co/hqZHOo/Screen_Shot_2018_07_24_at_17_42_21.png" width="100%"></p>

We then tested our system on multiple people and got great results. We can handle well as many people as fit in our camera view.
<p align="center"><img src="https://image.ibb.co/eH1Lb8/Screen_Shot_2018_07_24_at_17_42_33.png" width="100%"></p>

The touch detection also works well for as many people that fit the screen.
<p align="center"><img src="https://image.ibb.co/jiOoUT/Screen_Shot_2018_07_24_at_17_42_44.png" width="100%"></p>

# Video Demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=YOUTUBE_VIDEO_ID_HERE
" target="_blank"><img src="http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

# Running instructions


# Credits
This project was built in Haifa university image proccessing course.
This project was lead by Prof. Hagit Hel-or.
<b>Project members:</b>
- Itamar Shenhar : itamar8910@gmail.com
- Alon Melamud : alonmem@gmail.com
- Gil Maman : gil.maman.5@gmail.com
Contact for help :)
