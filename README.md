# 3D vision course final project
This project was built in Haifa university 3D vision course and lead by Prof. Hagit Hel-or.<br/>
In this project we aim to detect diffrent types of people interaction in 2 approaches.<br/>
<br/>
<b>Walkthrough:</b>
1. Problem definition <br/>
2. Input <br/>
3. First approach - 3D reconstruction<br/>
4. Second approach - 3D estimation <br/>
5. Output demo <br/>
6. Video demo <br/>
7. Running instructions <br/>
8. Credits

# Problem definition
<b>Problem:</b> Given 4 videos from 4 different cameras, <br/>
We aim to detect the type of interaction and when it happend.
<br/>
We currently support 4 types of interaction
- No interaction
- Touch
- Look
- Both

<b>Main challenge:</b> The system cannot assume anything about the scene <br/>
<p align="center"> <img src="https://image.ibb.co/e4AL9T/Screen_Shot_2018_07_24_at_11_25_30.png" width="50%"></p>

# Input
<p align="center"><img src="https://preview.ibb.co/eXVyw8/Screen_Shot_2018_07_24_at_11_42_42.png" width="50%"></p>
We receive as input: 4 RGB videos taken from different corners. 

# First approach - 3D reconstruction
<p align="center"><img src="https://preview.ibb.co/cmwSpT/Screen_Shot_2018_07_24_at_11_47_07.png" width="75%"></p>
<br/>
Our first approach was to reconstruct the 3D scene and to find the distance between each 2 "looking" vectors of people in the scene.
To do this, we first calibrate our 4 cameras.
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

This approach worked poorley. 
<br/>
This is because the many errors occurred while calibrating.
We wanted a different approach, that does not require calibration.
<p align="center"><img src="https://preview.ibb.co/n46VR8/Screen_Shot_2018_07_24_at_15_38_48.png" width="100%"></p>

# Second approach - 3D estimation
<p align="center"><img src="https://preview.ibb.co/jKXWYo/Screen_Shot_2018_07_24_at_15_46_24.png" width="100%"></p>
Instead of tring to reconstructe the 3D dimension scene, we tried to estimate it.
We used deep learning and Image proccssing algorithms in order to approximate the 3D scene.

<b>Our detection steps:</b><br/>
<b>1. Detect person using tinyFaces:</b> using deep learning classification(tinyFace model - CNN architecture) we can find the face dimensions of people in the scene with high accuracy even in very small and low resolution.
<p align="center"><img src="https://preview.ibb.co/iwLwzT/Screen_Shot_2018_07_24_at_15_49_31.png" width="100%"></p>

<b>2. Recognize person’s Identity:</b> Using deep learning one shot person recognition and a fall back to HSV color detection we can identify the people in the scene(With a few pictures taken from them priorly)
<p align="center"><img src="https://preview.ibb.co/itBato/Screen_Shot_2018_07_24_at_15_49_35.png" width="100%"></p>

<b>3. Find distance of person from camera:</b> Using the size of the person face and its body proportion extracted with openpose, we can estimate the user distance from the camera.
<p align="center"><img src="https://preview.ibb.co/nJPaR8/Screen_Shot_2018_07_24_at_15_49_39.png" width="100%"></p>

<b>4. Get looking direction:</b> Using gazer library which uses advanced deep learning and image proccessing techniques, we can find the gaze of the person. We also implemented a fall back heuristic based on the persons face - noise location to handle failure of gazer.
<p align="center"><img src="https://preview.ibb.co/d2Azm8/Screen_Shot_2018_07_24_at_15_49_43.png" width="100%"></p>

<b>5. Classify interaction:</b> finally, we can find the L2 distance between the two vectors from the peoples noise, and below a certain threshold alaram as interaction detection.
<p align="center"><img src="https://preview.ibb.co/dRUEKT/Screen_Shot_2018_07_24_at_15_56_02.png" width="100%"></p>

# Output demo
We were finally able to detect "looking" interaction. 
Whether two people are looking at each other, who are the people and how confident are we about the classification.
<p align="center"><img src="https://image.ibb.co/hqZHOo/Screen_Shot_2018_07_24_at_17_42_21.png" width="100%"></p>

We then tested our system on multiple people and got great results. We can handle well as many people as fit in our camera scene.
<p align="center"><img src="https://image.ibb.co/eH1Lb8/Screen_Shot_2018_07_24_at_17_42_33.png" width="100%"></p>

To handle touch detection, we leverage OpenPose skeleton. 
<br/>
For each camera, we check if the distance between the skeletons of the people are below a certian threshold. 
Ff this is correct for all cameras we can be sure its a touch.
After evaluation, this method works well and can detect touch for multiple people.
<p align="center"><img src="https://image.ibb.co/jiOoUT/Screen_Shot_2018_07_24_at_17_42_44.png" width="100%"></p>

# Video Demo
A short [video](https://www.youtube.com/watch?v=N-2G1_4ZBng) demonstrating our results.

# Running instructions


# Credits
This project was built in the University of Haifa, 3D vision course.<br/>
This project was lead by Prof. Hagit Hel-or.
<br/><br/><b>Project members:</b>
- Itamar Shenhar : itamar8910@gmail.com
- Alon Melamud : alonmem@gmail.com
- Gil Maman : gil.maman.5@gmail.com<br/>

<b>Fill free to contact us.</b>
