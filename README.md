# AI-VIRTUAL-ASSISTANT-AND-GESTURE-CONTROLLER
Control your Windows PC with hand gestures and talk to a local AI. Built with Python, OpenCV and MediaPipe: move the cursor, click, right-click and scroll with your hand, no extra hardware. An open palm launches a threaded voice assistant: speech recognition, a local Ollama DeepSeek-R1 model, and spoken replies. Lag-free, with a live FPS counter.

**AI Virtual Mouse & Assistant: Gesture Manual**

A quick guide to controlling Windows 11 with your hand and talking to your local AI assistant.
1. Quick start
1.	Make sure Ollama is running (it starts automatically after installation).
2.	Open Command Prompt in the folder with the script and run: python aivirtualmouseproject.py
3.	A camera window opens. Hold your hand about 40 to 70 cm from the webcam, palm facing the camera.
4.	Stay inside the purple rectangle on the video. That rectangle is your "touchpad": its edges map to the edges of your screen.
5.	Press Esc in the camera window to quit.
2. Gesture cheat sheet
Gesture	Fingers	Action
Point	Index only	Move the cursor
Peace sign + pinch	Index + Middle, tips touching	Left click
Three fingers, hold	Index + Middle + Ring	Right click
Fist, move hand up or down	All four fingers closed	Scroll
Open palm, hold	All five fingers	Voice assistant
The thumb is ignored by every gesture except the open palm, so keep it in whatever position is comfortable.
3. Gesture details
Move the cursor (Navigation)
•	Raise only your index finger; keep the other fingers curled.
•	The cursor follows your fingertip. Move inside the purple box to cover the whole screen.
•	To reach the taskbar: move your fingertip to the bottom edge of the purple box. The box ends high in the frame on purpose, so you do not need to lower your hand all the way down.
•	The cursor is smoothed: small movements are extra steady for precision, big sweeps are faster.
Left click
1.	Point at your target with the index finger and let the cursor settle.
2.	Raise the middle finger too (peace sign). The cursor stays where it is.
3.	Pinch the two fingertips together until they touch. The line between them turns green and you get one click.
4.	Pull the fingers apart again before clicking a second time. A short cooldown prevents accidental double clicks.
Right click
1.	Position the cursor with the index finger.
2.	Raise index, middle and ring fingers (pinky down) and hold still.
3.	The screen shows "RIGHT CLICK: hold... %". After about 0.6 seconds you get one right click.
4.	Lower the ring finger to reset. It will not repeat while you keep the same gesture.
Scroll (YouTube, web pages, documents)
1.	Put the cursor over the page you want to scroll (use the Navigation gesture).
2.	Close your hand into a fist. Hold it for a moment while "SCROLL: arming..." shows. Wherever your hand is at the end of this moment becomes the neutral point.
3.	Raise your hand above the neutral point to scroll up, lower it to scroll down.
4.	The further you move from the neutral point, the faster it scrolls. Return to the neutral line to stop.
5.	Open your hand or change gesture to finish.
If scrolling feels backwards, set SCROLL_DIRECTION = -1 in the script.
Voice assistant
1.	Show an open palm (all five fingers spread) and hold for about 0.7 seconds.
2.	Mouse control pauses and you hear "Listening..." followed by a short beep.
3.	After the beep, speak your question clearly.
4.	The text is sent to the local DeepSeek model through Ollama, and the answer is read aloud.
5.	The camera window shows what it heard ("You:") and the answer ("AI:").
6.	Mouse control returns automatically a couple of seconds after the answer.
No palm available? Press A in the camera window to start the assistant from the keyboard.
4. Keyboard shortcuts
•	H: show or hide the on-screen gesture cheat sheet
•	A: start the voice assistant manually
•	Esc: quit
Click the camera window first so it receives the key presses.
5. Tips for reliable tracking
•	Use good, even lighting. Avoid a bright window directly behind you.
•	A plain background works best.
•	Keep your hand fully visible, with the palm facing the camera.
•	Make clear, deliberate finger shapes. Half-bent fingers confuse the detector.
•	Move smoothly. A gesture must be recognised for two frames in a row before it takes effect, which filters out accidental detections.
•	Only one hand is tracked. Keep your other hand out of the frame.
6. Tuning settings
Open the script in a text editor and change these values near the top:
•	ACTIVE_BOTTOM (default 170): increase to reach the taskbar more easily, decrease if the cursor reaches the bottom too quickly.
•	ACTIVE_LEFT, ACTIVE_RIGHT, ACTIVE_TOP: size of the other sides of the purple box.
•	SMOOTH_MIN and SMOOTH_MAX: lower values give steadier but slower cursor movement.
•	PINCH_THRESHOLD_PX (default 35): raise it if clicks are hard to trigger.
•	RIGHT_CLICK_HOLD_SEC: how long to hold three fingers.
•	SCROLL_GAIN and SCROLL_DEADZONE_PX: scroll speed and how far you must move before it starts.
•	SCROLL_DIRECTION: 1 or -1 to flip scroll direction.
•	PALM_HOLD_SEC: how long to hold the open palm.
7. Testing the voice assistant
Run this command to test the speaker, microphone and Ollama one at a time, without the camera:
python aivirtualmouseproject.py --test
It prints PASS or FAIL for the microphone and for Ollama, and you will hear whether the speaker works.
8. Troubleshooting
•	Cursor cannot reach the bottom of the screen: increase ACTIVE_BOTTOM, and keep your fingertip inside the purple box.
•	Cursor is shaky: lower SMOOTH_MIN, improve lighting, rest your elbow on the desk.
•	Clicks trigger by accident: lower PINCH_THRESHOLD_PX, or raise CLICK_COOLDOWN_SEC.
•	Open palm is not detected: spread the fingers and thumb wide, face the palm to the camera, or use the A key.
•	"I can't access the microphone": check Windows Settings > Privacy & security > Microphone, and make sure PyAudio is installed.
•	"The speech service is unavailable": speech recognition uses Google and needs an internet connection.
•	"I can't reach the local AI model": start the Ollama app and run ollama pull deepseek-r1:1.5b. The first answer can take up to a minute while the model loads.
•	Nothing is spoken: check the volume and that Windows has a text-to-speech voice installed (Settings > Time & language > Speech).
