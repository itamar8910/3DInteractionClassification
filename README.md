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

