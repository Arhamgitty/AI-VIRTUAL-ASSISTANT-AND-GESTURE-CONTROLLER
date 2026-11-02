"""
=====================================================================================
 aivirtualmouseproject.py   (v2)
 Multimodal AI Virtual Assistant & Gesture Controller   (Windows 11)
=====================================================================================

 GESTURES
 --------
     INDEX only up ........................ Move the cursor (reaches the WHOLE screen,
                                            including the taskbar)
     INDEX + MIDDLE up, then PINCH ........ Left click
     INDEX + MIDDLE + RING up, HOLD ....... Right click (hold ~0.6 s)
     FIST (all fingers closed) ............ Scroll: raise hand = scroll up,
                                            lower hand = scroll down
     OPEN PALM (5 fingers), HOLD .......... Voice assistant (local Ollama LLM)

 KEYBOARD (click the camera window first)
 ----------------------------------------
     H ... show / hide the gesture cheat sheet
     A ... start the voice assistant manually
     Esc . quit

 COMMAND LINE
 ------------
     python aivirtualmouseproject.py            -> run the app
     python aivirtualmouseproject.py --test     -> test speaker, microphone and Ollama
                                                   WITHOUT the camera

 INSTALL (once, in Command Prompt)
 ---------------------------------
     pip install opencv-python numpy mediapipe pyautogui ollama SpeechRecognition pyttsx3 pyaudio
     ollama pull deepseek-r1:1.5b        (after installing Ollama from https://ollama.com)

 NOTE ON SAFETY
 --------------
     pyautogui's "corner abort" FAILSAFE is disabled because the gesture mapping
     must be able to reach screen corners (taskbar, Start button). Esc is the exit.
=====================================================================================
"""

# --------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------
import argparse      # Command-line flag --test
import math          # Euclidean distance (hypot) and sign helpers
import queue         # Thread-safe FIFO that feeds the text-to-speech worker thread
import re            # Regular expressions to strip <think> tags and markdown from LLM text
import sys           # Clean process exit with an error code when startup fails
import threading     # Background threads: speech recognition, LLM call, TTS playback
import time          # Timestamps for FPS, debounce, cooldowns and hold timers

import cv2           # OpenCV: webcam capture, overlays, display window
import numpy as np   # np.interp performs the camera-space -> screen-space mapping
import pyautogui     # Moves the physical Windows cursor, clicks and scrolls
import pyttsx3       # Offline text-to-speech (Windows SAPI5 voices)
import speech_recognition as sr   # Microphone capture + Google Web Speech transcription
import ollama        # Python client for the local Ollama server
import mediapipe as mp            # Hand landmark detection (21 landmarks per hand)

try:
    import winsound  # Windows-only standard-library beep (used as a "speak now" cue)
except ImportError:  # Non-Windows platforms: simply skip the beep
    winsound = None

# --------------------------------------------------------------------------------------
# GLOBAL CONFIGURATION  (tweak these without touching program logic)
# --------------------------------------------------------------------------------------
# --- Camera ---
CAM_INDEX = 0                      # 0 = default laptop webcam
CAM_WIDTH, CAM_HEIGHT = 640, 480   # Capture resolution (small = fast)
WINDOW_NAME = "AI Virtual Mouse & Assistant  (H = help, Esc = quit)"

# --- Active box: the part of the camera frame that is mapped to the WHOLE screen ---
# Values are margins in webcam pixels. The index fingertip travelling across this box
# sweeps the cursor across the entire monitor.
#   ACTIVE_BOTTOM is deliberately LARGE: most people hold their hand in the upper part of
#   the camera view, so the box ends high up. Reaching its bottom edge puts the cursor on
#   the taskbar. If you still can't reach the taskbar, INCREASE ACTIVE_BOTTOM (e.g. 200).
#   If the cursor reaches the taskbar too easily / jumps, DECREASE it.
ACTIVE_LEFT = 60
ACTIVE_RIGHT = 60
ACTIVE_TOP = 40
ACTIVE_BOTTOM = 170

# --- Cursor smoothing (distance-adaptive exponential smoothing) ---
# Slow, small movements use SMOOTH_MIN (very steady, kills jitter).
# Fast, long movements use SMOOTH_MAX (cursor keeps up with your hand).
SMOOTH_MIN = 0.15
SMOOTH_MAX = 0.55

# --- Stability ---
GESTURE_CONFIRM_FRAMES = 2         # A new gesture must be seen this many frames in a row

# --- Left click ---
PINCH_THRESHOLD_PX = 35            # Index-tip <-> middle-tip distance (webcam px) = pinched
CLICK_COOLDOWN_SEC = 0.40          # Minimum time between any two clicks (anti double-click)

# --- Right click ---
RIGHT_CLICK_HOLD_SEC = 0.60        # Hold 3 fingers up this long to right-click

# --- Scrolling (fist gesture, "joystick" style) ---
SCROLL_ARM_SEC = 0.30              # Hold the fist this long before scrolling activates
SCROLL_DEADZONE_PX = 25            # Hand must move this far from its start point to scroll
SCROLL_GAIN = 3.0                  # Scroll strength per pixel beyond the dead zone
SCROLL_MAX_PER_TICK = 240          # Cap per scroll event (Windows: 120 = one wheel notch)
SCROLL_INTERVAL_SEC = 0.04         # Time between scroll events (~25 per second)
SCROLL_DIRECTION = 1               # 1 = hand up scrolls up.  -1 = inverted (touch-screen style)

# --- Assistant behaviour ---
PALM_HOLD_SEC = 0.70               # Open palm must be held this long to launch the assistant
ASSISTANT_COOLDOWN_SEC = 2.0       # Dead time after a session before it can re-trigger
LLM_MODEL = "deepseek-r1:1.5b"             # Local Ollama model name
OLLAMA_HOST = "http://localhost:11434"     # Default local Ollama endpoint
OLLAMA_TIMEOUT_SEC = 120           # Max wait for one LLM answer (first call loads the model)
LISTEN_START_TIMEOUT_SEC = 6       # Max seconds to wait for the user to START speaking
LISTEN_PHRASE_LIMIT_SEC = 12       # Max seconds of a single spoken phrase
MAX_SPOKEN_CHARS = 700             # Safety cap so a very long answer is not read forever
SYSTEM_PROMPT = (                  # Keeps answers short, speakable and free of markdown
    "You are a concise voice assistant. Answer in one to three short plain sentences. "
    "Do not use markdown, lists, emojis or special symbols."
)

# --- MediaPipe landmark indices ---
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# --- Overlay colours (BGR) ---
COL_GREEN = (80, 220, 80)
COL_BLUE = (255, 160, 40)
COL_YELLOW = (0, 220, 255)
COL_RED = (60, 60, 255)
COL_WHITE = (255, 255, 255)
COL_GREY = (170, 170, 170)
COL_PURPLE = (255, 0, 255)

# --- Text shown by the H-key cheat sheet ---
HELP_LINES = [
    ("INDEX up only", "Move cursor"),
    ("INDEX+MIDDLE, pinch", "Left click"),
    ("INDEX+MIDDLE+RING, hold", "Right click"),
    ("FIST, hand up / down", "Scroll up / down"),
    ("OPEN PALM, hold", "Voice assistant"),
    ("Keys: H help  A assistant", "Esc quit"),
]


# --------------------------------------------------------------------------------------
# WINDOWS 11 DPI AWARENESS
# --------------------------------------------------------------------------------------
def enable_dpi_awareness():
    """
    On Windows 11 laptops with display scaling (125%, 150%...), Python may see a
    *scaled-down* screen size, so the cursor would not reach the real screen edges
    (another reason the taskbar can be unreachable). Declaring the process DPI-aware
    makes pyautogui report the true pixel size.
    """
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # 2 = per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()    # Older fallback API
        except Exception:
            pass   # Non-fatal


# --------------------------------------------------------------------------------------
# LLM TEXT CLEANING  (DeepSeek-R1 optimisation)
# --------------------------------------------------------------------------------------
def clean_llm_response(raw_text: str) -> str:
    """
    DeepSeek-R1 emits its chain-of-thought inside <think> ... </think> before the real
    answer. Reading that aloud is useless, so we:
        1. Remove every complete <think>...</think> block (multi-line, case-insensitive).
        2. Remove an UNCLOSED <think> block (model cut off mid-thought).
        3. Remove any stray <think> / </think> tags.
        4. Strip markdown symbols that a speech engine would pronounce literally.
        5. Collapse whitespace.
    """
    if not raw_text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_`#>~|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def beep(freq=880, ms=150):
    """Short 'speak now' cue so the user knows the microphone is about to open."""
    try:
        if winsound is not None:
            winsound.Beep(freq, ms)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# TEXT-TO-SPEECH WORKER THREAD
# --------------------------------------------------------------------------------------
class TTSWorker(threading.Thread):
    """
    A dedicated thread that does ALL speech output.

    Why a dedicated thread?
      * engine.runAndWait() blocks until speech finishes; on the main thread it would
        freeze the webcam loop.
      * SAPI5 (COM) must be used from the thread that initialised it, so every request
        is funnelled through a queue into this one thread.
      * A FRESH engine is created for every sentence. This avoids the well-known
        pyttsx3 bug where the voice goes silent after the first utterance.
    """

    def __init__(self):
        super().__init__(daemon=True, name="TTSWorker")
        self._queue = queue.Queue()      # Items: (text, done_event_or_None) or None (= quit)

    def run(self):
        com_initialised = False
        try:
            # Windows COM must be initialised in every thread that uses SAPI5 voices.
            import comtypes
            comtypes.CoInitialize()
            com_initialised = True
        except Exception:
            pass   # comtypes missing / already initialised: pyttsx3 may still work

        while True:
            item = self._queue.get()
            if item is None:             # Sentinel -> shut the thread down
                break
            text, done_event = item
            engine = None
            try:
                if text:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 175)    # Words per minute
                    engine.setProperty("volume", 1.0)  # 0.0 - 1.0
                    engine.say(text)
                    engine.runAndWait()  # Blocks THIS worker thread only, not the camera
            except Exception as exc:
                print(f"[TTS] Speech failed: {exc}")
            finally:
                try:
                    if engine is not None:
                        engine.stop()
                except Exception:
                    pass
                del engine
                if done_event is not None:
                    done_event.set()     # Wake any thread waiting for this speech to finish

        if com_initialised:
            try:
                import comtypes
                comtypes.CoUninitialize()
            except Exception:
                pass

    def say(self, text: str, wait: bool = False, timeout: float = 120.0):
        """
        Queue text for speaking.
          wait=False -> fire and forget.
          wait=True  -> block the CALLING (background) thread until speech completes.
        Never call with wait=True from the camera thread.
        """
        done_event = threading.Event() if wait else None
        self._queue.put((text, done_event))
        if wait and done_event is not None:
            done_event.wait(timeout=timeout)

    def shutdown(self):
        """Ask the worker to exit after finishing queued items."""
        self._queue.put(None)


# --------------------------------------------------------------------------------------
# VOICE ASSISTANT (SPEECH -> TEXT -> OLLAMA -> SPEECH) ON A BACKGROUND THREAD
# --------------------------------------------------------------------------------------
class VoiceAssistant:
    """
    One complete assistant interaction, run in a worker thread:
        say "Listening..." -> beep -> record mic -> Google STT -> Ollama -> speak reply
    The camera loop only reads `busy` and the snapshot() text, so it never blocks.
    """

    def __init__(self, tts: TTSWorker):
        self.tts = tts
        self.busy = threading.Event()          # Set while a session is running
        self._lock = threading.Lock()          # Guards the text fields below
        self._status = "Idle"
        self._last_user_text = ""
        self._last_reply = ""
        self.last_finished = 0.0               # Used for the re-trigger cooldown

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True   # Adapts to room noise
        self.recognizer.pause_threshold = 0.9             # Silence (s) that ends a phrase

        # Explicit timeout so a hung server can never keep the thread alive forever.
        self.client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT_SEC)

    # ---- thread-safe state accessors ------------------------------------------------
    def _set(self, status=None, user_text=None, reply=None):
        with self._lock:
            if status is not None:
                self._status = status
            if user_text is not None:
                self._last_user_text = user_text
            if reply is not None:
                self._last_reply = reply

    def snapshot(self):
        """Return (status, last_user_text, last_reply) atomically for drawing."""
        with self._lock:
            return self._status, self._last_user_text, self._last_reply

    # ---- public trigger ---------------------------------------------------------------
    def trigger(self) -> bool:
        """Start a session on a daemon thread. Returns False if one is already running."""
        if self.busy.is_set():
            return False
        self.busy.set()
        threading.Thread(target=self._run_session, daemon=True, name="AssistantSession").start()
        return True

    # ---- the actual pipeline ----------------------------------------------------------
    def _run_session(self):
        try:
            # 1) Audible cue. wait=True so the microphone does not record our own voice.
            self._set(status="Listening...", user_text="", reply="")
            self.tts.say("Listening...", wait=True)
            beep()                                   # "speak now" signal

            # 2) Capture + transcribe speech.
            user_text = self._listen_and_transcribe()
            if not user_text:
                return   # The failure was already spoken/displayed

            self._set(status="Thinking...", user_text=user_text)
            print(f"[You] {user_text}")

            # 3) Ask the local LLM.
            reply = self._ask_ollama(user_text)
            if not reply:
                return

            # 4) Speak the cleaned answer.
            self._set(status="Speaking...", reply=reply)
            print(f"[Assistant] {reply}")
            self.tts.say(reply, wait=True)

        except Exception as exc:   # Last line of defence for the thread
            print(f"[Assistant] Unexpected error: {exc}")
            self._set(status="Assistant error")
        finally:
            self.last_finished = time.time()
            self._set(status="Idle")
            self.busy.clear()      # Always release, even after failures

    def _listen_and_transcribe(self) -> str:
        """Record from the default mic and transcribe with Google. Returns "" on failure."""
        try:
            with sr.Microphone() as source:
                # Sample ambient noise so the energy threshold fits this room.
                self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                audio = self.recognizer.listen(
                    source,
                    timeout=LISTEN_START_TIMEOUT_SEC,
                    phrase_time_limit=LISTEN_PHRASE_LIMIT_SEC,
                )
        except sr.WaitTimeoutError:
            self._fail("I didn't hear anything.")
            return ""
        except (OSError, AttributeError) as exc:
            # OSError: no mic / device busy / Windows privacy settings block the mic.
            # AttributeError: PyAudio is missing.
            print(f"[Mic] Microphone unavailable: {exc}")
            self._fail("I can't access the microphone.")
            return ""
        except Exception as exc:
            print(f"[Mic] Recording error: {exc}")
            self._fail("There was a microphone problem.")
            return ""

        # Transcribe OUTSIDE the `with` block so the mic is released first.
        try:
            self._set(status="Transcribing...")
            return self.recognizer.recognize_google(audio).strip()
        except sr.UnknownValueError:
            self._fail("Sorry, I couldn't understand that.")    # noise / mumbling
        except sr.RequestError as exc:
            print(f"[STT] Google speech service error: {exc}")  # offline / quota
            self._fail("The speech service is unavailable. Check your internet connection.")
        except Exception as exc:
            print(f"[STT] Transcription error: {exc}")
            self._fail("Speech recognition failed.")
        return ""

    def _ask_ollama(self, prompt: str) -> str:
        """Send the prompt to local Ollama and return a speakable answer ("" on failure)."""
        try:
            response = self.client.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response["message"]["content"]
        except ollama.ResponseError as exc:
            # Server reachable but returned an error (most often: model not pulled).
            print(f"[Ollama] Response error ({getattr(exc, 'status_code', '?')}): {exc}")
            self._fail(f"The model {LLM_MODEL} is not available. Please pull it with Ollama.")
            return ""
        except Exception as exc:
            # Connection refused / server not running / timeout / malformed reply.
            print(f"[Ollama] Request failed: {exc}")
            self._fail("I can't reach the local AI model. Is Ollama running?")
            return ""

        answer = clean_llm_response(raw)   # Strip <think> blocks + markdown
        if not answer:
            self._fail("The model did not return a final answer.")
            return ""

        if len(answer) > MAX_SPOKEN_CHARS:   # Trim at a sentence boundary when possible
            cut = answer[:MAX_SPOKEN_CHARS]
            last_stop = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
            answer = cut[: last_stop + 1] if last_stop > 100 else cut
        return answer

    def _fail(self, spoken_message: str):
        """Show + speak a short error so the user knows what went wrong."""
        self._set(status=spoken_message)
        print(f"[Assistant] {spoken_message}")
        self.tts.say(spoken_message, wait=True)


# --------------------------------------------------------------------------------------
# SELF-TEST (python aivirtualmouseproject.py --test)
# --------------------------------------------------------------------------------------
def run_selftest():
    """
    Tests the three voice-assistant parts one by one, WITHOUT the camera, so you can
    see exactly which part (speaker / microphone / Ollama) is failing.
    """
    print("=" * 60)
    print(" VOICE ASSISTANT SELF-TEST")
    print("=" * 60)

    tts = TTSWorker()
    tts.start()
    assistant = VoiceAssistant(tts)

    # ---- Test 1: speaker ---------------------------------------------------------------
    print("\n[1/3] Speaker / text-to-speech")
    tts.say("Text to speech is working.", wait=True)
    print("      You should have HEARD a voice. If not: check volume and Windows voices.")

    # ---- Test 2: microphone + Google speech recognition --------------------------------
    print("\n[2/3] Microphone + speech recognition")
    try:
        names = sr.Microphone.list_microphone_names()
        print(f"      Microphones found: {len(names)}")
        for i, name in enumerate(names[:6]):
            print(f"        [{i}] {name}")
    except Exception as exc:
        print(f"      Could not list microphones (is PyAudio installed?): {exc}")
    print("      After the beep, say something like 'what is the capital of France'...")
    tts.say("Listening...", wait=True)
    beep()
    heard = assistant._listen_and_transcribe()
    print(f"      Transcript: {heard!r}")
    mic_ok = bool(heard)

    # ---- Test 3: Ollama --------------------------------------------------------------
    print("\n[3/3] Ollama + DeepSeek model (first call can take 10-60 s while the model loads)")
    reply = assistant._ask_ollama("Say hello in one short sentence.")
    print(f"      Reply: {reply!r}")
    llm_ok = bool(reply)
    if llm_ok:
        tts.say(reply, wait=True)

    print("\n" + "=" * 60)
    print(f" Speaker : (you decide - did you hear it?)")
    print(f" Mic/STT : {'PASS' if mic_ok else 'FAIL'}")
    print(f" Ollama  : {'PASS' if llm_ok else 'FAIL'}")
    print("=" * 60)
    tts.shutdown()
    time.sleep(0.5)


# --------------------------------------------------------------------------------------
# HAND GEOMETRY HELPERS
# --------------------------------------------------------------------------------------
def dist(a, b) -> float:
    """Euclidean distance between two [id, x, y] landmark entries (pixel units)."""
    return math.hypot(a[1] - b[1], a[2] - b[2])


def fingers_up(lm_list):
    """
    Return [thumb, index, middle, ring, pinky] with 1 = extended, 0 = folded.

    Rotation-independent tests (the hand need not be perfectly upright):
      * Index/Middle/Ring/Pinky: extended when the TIP is farther from the WRIST than
        the PIP joint. A curled finger brings the tip back toward the palm.
      * Thumb: moves sideways, so compare distances to the pinky knuckle (MCP): an
        extended thumb tip is clearly farther from it than the thumb's IP joint.
    """
    states = []

    thumb_extended = dist(lm_list[THUMB_TIP], lm_list[PINKY_MCP]) > \
        1.10 * dist(lm_list[THUMB_IP], lm_list[PINKY_MCP])
    states.append(1 if thumb_extended else 0)

    for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                     (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)):
        extended = dist(lm_list[tip], lm_list[WRIST]) > dist(lm_list[pip], lm_list[WRIST])
        states.append(1 if extended else 0)

    return states


def classify_gesture(fingers) -> str:
    """
    Convert the finger states into ONE gesture name. The thumb is ignored by every
    gesture except OPEN PALM (needs all five), which keeps gestures easy to form.
        PALM   all five up                        -> voice assistant
        NAV    index only                         -> move cursor
        CLICK  index + middle                     -> left click on pinch
        RIGHT  index + middle + ring              -> right click on hold
        FIST   index, middle, ring, pinky all down-> scroll
        IDLE   anything else                      -> do nothing
    """
    thumb, index, middle, ring, pinky = fingers
    if sum(fingers) == 5:
        return "PALM"
    if index and not middle and not ring and not pinky:
        return "NAV"
    if index and middle and not ring and not pinky:
        return "CLICK"
    if index and middle and ring and not pinky:
        return "RIGHT"
    if not index and not middle and not ring and not pinky:
        return "FIST"
    return "IDLE"


# --------------------------------------------------------------------------------------
# DRAWING HELPERS
# --------------------------------------------------------------------------------------
def put_text(img, text, org, color=COL_WHITE, scale=0.6, thickness=2):
    """Draw text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def shorten(text: str, max_chars: int) -> str:
    """Trim long strings so they fit inside the camera overlay."""
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def draw_help(img):
    """Semi-transparent cheat sheet toggled with the H key."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = 20, 80, w - 20, 80 + 28 * len(HELP_LINES) + 16
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)     # Blend for transparency
    for i, (gesture, action) in enumerate(HELP_LINES):
        y = y0 + 28 + 28 * i
        put_text(img, gesture, (x0 + 10, y), COL_YELLOW, 0.5, 1)
        put_text(img, action, (x0 + 330, y), COL_WHITE, 0.5, 1)


# --------------------------------------------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------------------------------------------
def main():
    # ---- one-time system setup ---------------------------------------------------------
    enable_dpi_awareness()
    pyautogui.FAILSAFE = False   # Corners must be reachable (taskbar / Start button)
    pyautogui.PAUSE = 0          # Remove pyautogui's default 0.1 s delay after each call
    screen_w, screen_h = pyautogui.size()   # TRUE screen size in pixels

    if not hasattr(mp, "solutions"):
        print("This MediaPipe build lacks the legacy `mp.solutions` API. "
              "Install a compatible version:  pip install \"mediapipe<0.10.30\"")
        sys.exit(1)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # model_complexity=0 = lightweight model -> highest FPS on laptop CPUs.
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # ---- background services ------------------------------------------------------------
    tts = TTSWorker()
    tts.start()
    assistant = VoiceAssistant(tts)

    # ---- open the webcam ----------------------------------------------------------------
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)   # DirectShow opens fastest on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                # Newest frame only -> lower latency
    if not cap.isOpened():
        print("ERROR: Could not open the webcam. Close other apps using it and check "
              "Settings > Privacy & security > Camera.")
        tts.shutdown()
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    # ---- runtime state ------------------------------------------------------------------
    prev_time = time.time()
    fps = 0.0
    px, py = pyautogui.position()
    cur_x, cur_y = float(px), float(py)    # Smoothed cursor position
    last_click_time = 0.0                  # Debounce timer shared by left + right click
    last_scroll_time = 0.0

    gesture = "NONE"                       # Confirmed (stable) gesture
    candidate, cand_count = "NONE", 0      # Gesture currently being confirmed

    palm_start = None                      # Open-palm hold timer
    pinch_latched = False                  # True while fingers stay pinched (1 pinch = 1 click)
    rc_start = None                        # Right-click hold timer
    rc_latched = False                     # True after a right click until gesture changes
    fist_start = None                      # Fist hold timer (scroll arming)
    scroll_anchor_y = 0.0                  # Hand height when scrolling armed (the "neutral")
    scroll_smooth_y = 0.0                  # Smoothed current hand height

    show_help = False
    mode_text, mode_color = "NO HAND", COL_GREY
    failed_reads = 0

    print("Running. Show your hand to the camera. H = help, A = assistant, Esc = quit.")

    try:
        # ================================ MAIN LOOP ======================================
        while True:
            # ---- 1) Acquire frame -------------------------------------------------------
            ok, img = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads > 30:
                    print("ERROR: Camera stopped delivering frames.")
                    break
                continue
            failed_reads = 0

            img = cv2.flip(img, 1)      # Mirror fix: hand right = cursor right
            frame_h, frame_w = img.shape[:2]

            # Active box edges (computed per frame in case the frame size differs).
            box_l, box_r = ACTIVE_LEFT, frame_w - ACTIVE_RIGHT
            box_t, box_b = ACTIVE_TOP, frame_h - ACTIVE_BOTTOM

            # ---- 2) Hand landmark detection ---------------------------------------------
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            lm_list = []                # [[id, x_px, y_px], ...] for the first hand
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                for lm_id, lm in enumerate(hand_lms.landmark):
                    lm_list.append([lm_id, int(lm.x * frame_w), int(lm.y * frame_h)])
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            cv2.rectangle(img, (box_l, box_t), (box_r, box_b), COL_PURPLE, 2)   # Active box

            assistant_busy = assistant.busy.is_set()
            now = time.time()

            # ---- 3) Gesture classification + stabilisation ------------------------------
            # Safeguard: only touch landmarks when a hand is present (prevents IndexError
            # when the hand leaves the frame).
            fingers = [0, 0, 0, 0, 0]
            if len(lm_list) != 0:
                fingers = fingers_up(lm_list)
                raw_gesture = classify_gesture(fingers)

                # A gesture must be seen N frames in a row before we accept it. This removes
                # one-frame misdetections that would cause stray clicks or scrolls.
                if raw_gesture == candidate:
                    cand_count += 1
                else:
                    candidate, cand_count = raw_gesture, 1
                new_gesture = candidate if cand_count >= GESTURE_CONFIRM_FRAMES else gesture
            else:
                candidate, cand_count = "NONE", 0
                new_gesture = "NONE"

            # On any gesture change, reset every per-gesture timer / latch.
            if new_gesture != gesture:
                palm_start = None
                pinch_latched = False
                rc_start = None
                rc_latched = False
                fist_start = None
                if new_gesture == "NAV":
                    # Re-sync the smoothed position with the real cursor so it glides
                    # from where it actually is (no sudden jumps).
                    px, py = pyautogui.position()
                    cur_x, cur_y = float(px), float(py)
                gesture = new_gesture

            # ---- 4) Act on the gesture --------------------------------------------------
            if len(lm_list) != 0:
                x1, y1 = lm_list[INDEX_TIP][1], lm_list[INDEX_TIP][2]
                x2, y2 = lm_list[MIDDLE_TIP][1], lm_list[MIDDLE_TIP][2]

                if assistant_busy:
                    # Assistant running: mouse control is halted.
                    mode_text, mode_color = "ASSISTANT ACTIVE (mouse paused)", COL_YELLOW
                    palm_start = None

                # ------------------------- OPEN PALM -> ASSISTANT -----------------------
                elif gesture == "PALM":
                    if palm_start is None:
                        palm_start = now
                    held = now - palm_start
                    cooldown_left = ASSISTANT_COOLDOWN_SEC - (now - assistant.last_finished)

                    if cooldown_left > 0:
                        mode_text, mode_color = f"ASSISTANT COOLDOWN {cooldown_left:.1f}s", COL_GREY
                    elif held >= PALM_HOLD_SEC:
                        if assistant.trigger():       # Spawns background thread, returns at once
                            mode_text, mode_color = "ASSISTANT STARTING", COL_YELLOW
                        palm_start = None
                    else:
                        mode_text = f"OPEN PALM: hold... {int(100 * held / PALM_HOLD_SEC)}%"
                        mode_color = COL_YELLOW

                # ------------------------- NAVIGATION -----------------------------------
                elif gesture == "NAV":
                    mode_text, mode_color = "NAVIGATION", COL_GREEN

                    # Camera box -> full screen. np.interp clamps outside the range, so
                    # reaching the box edge means reaching the screen edge (taskbar!).
                    target_x = float(np.interp(x1, (box_l, box_r), (0, screen_w)))
                    target_y = float(np.interp(y1, (box_t, box_b), (0, screen_h)))

                    # Distance-adaptive exponential smoothing:
                    #   new = old + alpha * (target - old)
                    # alpha is small for tiny movements (kills jitter) and larger for big
                    # sweeps (so the cursor keeps up with a fast hand).
                    dx, dy = target_x - cur_x, target_y - cur_y
                    d = math.hypot(dx, dy)
                    alpha = float(np.interp(d, (4, 200), (SMOOTH_MIN, SMOOTH_MAX)))
                    if d > 1.5:                       # Ignore sub-pixel noise completely
                        cur_x += dx * alpha
                        cur_y += dy * alpha

                    move_x = int(min(max(cur_x, 0), screen_w - 1))
                    move_y = int(min(max(cur_y, 0), screen_h - 1))
                    try:
                        pyautogui.moveTo(move_x, move_y)    # Moves the real Windows cursor
                    except Exception as exc:
                        print(f"[Mouse] moveTo failed: {exc}")

                    cv2.circle(img, (x1, y1), 12, COL_GREEN, cv2.FILLED)

                # ------------------------- LEFT CLICK -----------------------------------
                elif gesture == "CLICK":
                    pinch_distance = math.hypot(x2 - x1, y2 - y1)
                    mid_pt = ((x1 + x2) // 2, (y1 + y2) // 2)
                    pinched = pinch_distance < PINCH_THRESHOLD_PX

                    cv2.line(img, (x1, y1), (x2, y2), COL_GREEN if pinched else COL_BLUE, 3)
                    cv2.circle(img, (x1, y1), 9, COL_BLUE, cv2.FILLED)
                    cv2.circle(img, (x2, y2), 9, COL_BLUE, cv2.FILLED)
                    mode_text, mode_color = f"CLICK MODE  d={int(pinch_distance)}px", COL_BLUE

                    if pinched:
                        # Click only on the pinch EDGE (not while held) and after the
                        # cooldown: together these prevent accidental double clicks.
                        if (not pinch_latched) and (now - last_click_time) > CLICK_COOLDOWN_SEC:
                            try:
                                pyautogui.click()
                            except Exception as exc:
                                print(f"[Mouse] click failed: {exc}")
                            last_click_time = now
                            pinch_latched = True
                        cv2.circle(img, mid_pt, 14, COL_RED, cv2.FILLED)
                        mode_text, mode_color = "LEFT CLICK!", COL_RED
                    else:
                        pinch_latched = False     # Fingers apart -> re-arm

                # ------------------------- RIGHT CLICK ----------------------------------
                elif gesture == "RIGHT":
                    for tip in (INDEX_TIP, MIDDLE_TIP, RING_TIP):
                        cv2.circle(img, (lm_list[tip][1], lm_list[tip][2]), 9, COL_BLUE, cv2.FILLED)

                    if rc_latched:
                        # Already clicked: wait until the user changes gesture.
                        mode_text, mode_color = "RIGHT CLICKED - lower ring finger", COL_RED
                    else:
                        if rc_start is None:
                            rc_start = now
                        held = now - rc_start
                        if held >= RIGHT_CLICK_HOLD_SEC:
                            if (now - last_click_time) > CLICK_COOLDOWN_SEC:
                                try:
                                    pyautogui.click(button="right")   # Native right click
                                except Exception as exc:
                                    print(f"[Mouse] right click failed: {exc}")
                                last_click_time = now
                            rc_latched = True
                            mode_text, mode_color = "RIGHT CLICK!", COL_RED
                        else:
                            mode_text = f"RIGHT CLICK: hold... {int(100 * held / RIGHT_CLICK_HOLD_SEC)}%"
                            mode_color = COL_BLUE

                # ------------------------- SCROLL (FIST) --------------------------------
                elif gesture == "FIST":
                    hand_x = lm_list[MIDDLE_MCP][1]
                    hand_y = float(lm_list[MIDDLE_MCP][2])    # Knuckle height = scroll input

                    if fist_start is None:
                        fist_start = now

                    if now - fist_start < SCROLL_ARM_SEC:
                        # ARMING: keep re-setting the neutral point while the hand settles,
                        # so closing the fist does not itself cause a scroll.
                        scroll_anchor_y = hand_y
                        scroll_smooth_y = hand_y
                        mode_text, mode_color = "SCROLL: arming...", COL_YELLOW
                    else:
                        # Smooth the hand height, then measure distance from the neutral point.
                        scroll_smooth_y += (hand_y - scroll_smooth_y) * 0.4
                        offset = scroll_anchor_y - scroll_smooth_y   # + = hand raised

                        # Visual guide: neutral line, dead-zone band, current hand marker.
                        ay = int(scroll_anchor_y)
                        cv2.line(img, (hand_x - 60, ay), (hand_x + 60, ay), COL_WHITE, 1)
                        cv2.line(img, (hand_x - 40, ay - SCROLL_DEADZONE_PX), (hand_x + 40, ay - SCROLL_DEADZONE_PX), COL_GREY, 1)
                        cv2.line(img, (hand_x - 40, ay + SCROLL_DEADZONE_PX), (hand_x + 40, ay + SCROLL_DEADZONE_PX), COL_GREY, 1)
                        cv2.circle(img, (hand_x, int(scroll_smooth_y)), 10, COL_YELLOW, cv2.FILLED)

                        if abs(offset) > SCROLL_DEADZONE_PX:
                            # Further from neutral = faster scrolling (joystick behaviour).
                            magnitude = min((abs(offset) - SCROLL_DEADZONE_PX) * SCROLL_GAIN,
                                            SCROLL_MAX_PER_TICK)
                            amount = int(math.copysign(magnitude, offset)) * SCROLL_DIRECTION
                            direction_word = "UP" if amount > 0 else "DOWN"
                            mode_text, mode_color = f"SCROLLING {direction_word}", COL_GREEN
                            if amount != 0 and (now - last_scroll_time) >= SCROLL_INTERVAL_SEC:
                                try:
                                    pyautogui.scroll(amount)   # + = up, - = down
                                except Exception as exc:
                                    print(f"[Mouse] scroll failed: {exc}")
                                last_scroll_time = now
                        else:
                            mode_text, mode_color = "SCROLL: ready (move hand up/down)", COL_YELLOW

                # ------------------------- ANYTHING ELSE --------------------------------
                else:
                    mode_text, mode_color = "IDLE (unrecognised gesture)", COL_GREY

                put_text(img, f"Fingers {fingers}  gesture={gesture}", (10, frame_h - 12), COL_GREY, 0.5, 1)

            else:
                # No hand visible.
                if assistant_busy:
                    mode_text, mode_color = "ASSISTANT ACTIVE (mouse paused)", COL_YELLOW
                else:
                    mode_text, mode_color = "NO HAND", COL_GREY

            # ---- 5) FPS counter (top-left) ----------------------------------------------
            now_t = time.time()
            inst_fps = 1.0 / max(now_t - prev_time, 1e-6)
            prev_time = now_t
            fps = inst_fps if fps == 0 else 0.9 * fps + 0.1 * inst_fps   # Running average
            put_text(img, f"FPS: {int(fps)}", (10, 28), COL_GREEN, 0.8, 2)

            # ---- 6) Mode + assistant status overlay -------------------------------------
            put_text(img, mode_text, (10, 58), mode_color, 0.6, 2)
            put_text(img, "H = help", (frame_w - 90, 24), COL_GREY, 0.5, 1)

            status, user_text, reply_text = assistant.snapshot()
            if status != "Idle" or user_text or reply_text:
                put_text(img, f"Assistant: {shorten(status, 60)}", (10, frame_h - 70), COL_YELLOW, 0.5, 1)
                if user_text:
                    put_text(img, f"You: {shorten(user_text, 70)}", (10, frame_h - 52), COL_WHITE, 0.5, 1)
                if reply_text:
                    put_text(img, f"AI: {shorten(reply_text, 70)}", (10, frame_h - 34), COL_GREEN, 0.5, 1)

            if show_help:
                draw_help(img)

            # ---- 7) Display + keyboard --------------------------------------------------
            cv2.imshow(WINDOW_NAME, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                                   # Esc -> clean exit
                print("Esc pressed. Shutting down...")
                break
            elif key in (ord("h"), ord("H")):               # Toggle cheat sheet
                show_help = not show_help
            elif key in (ord("a"), ord("A")):               # Manual assistant trigger
                assistant.trigger()

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break                                       # Window closed with the [X]

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # Guaranteed cleanup no matter how we exit.
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        tts.shutdown()
        print("Goodbye.")


# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Virtual Mouse & Voice Assistant")
    parser.add_argument("--test", action="store_true",
                        help="Test speaker, microphone and Ollama without opening the camera")
    args = parser.parse_args()

    if args.test:
        run_selftest()
    else:
        main()
