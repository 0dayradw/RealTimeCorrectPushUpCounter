# Real-time CORRECT Form Push-Up Counter
This project helps track and improve push-up form by detecting repetitions and correctness in real-time.  
I built it to learn computer vision - and also to improve my own push-ups (I'm using this program daily)

## Notes
- The program counts how many **correct push-ups** you do in real time  
- A push-up is considered correct if:
  - the **elbow angle** goes below 90° when going down
  - and it **exceeds 145/155°** when going up
- At each step the program **tells you when to go down / up**, so that you are on the correct pace.
- You could also make the push-ups harder by adjusting 'valid_frames' value (from 1 to  5 / 6 or 10). 
  - The higher the value, the **longer you must hold** the “up” or “down” position.

## Tools Used

- **Python**
- **MediaPipe** - i'm using only 3 * 2 keypoints : left/right shoulder, elbow and wrist
- **OpenCV** - camera input
  - note that my camera is only 1280x720, if yours is 1920x1080 adjust the resolution : CAMERA_WIDTH / CAMERA_HEIGHT


## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/0dayradw/RealTimeCorrectPushUpCounter.git
   cd RealTimeCorrectPushUpCounter  
   ```
2. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```
3. Run the program and start doing push-ups )))
   ```bash
   python3 main.py
   ```
   
