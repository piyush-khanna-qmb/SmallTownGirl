"""
SmallTownGirl — Hand Gesture based PC control (two-hand horns toggle + momentum flick).

CONTROL (either hand): make a "horns" pose -- index + pinky extended, middle &
ring curled -- and hold it ~1s to TOGGLE scroll-control on/off. Hold horns again
for ~1s to turn it back off. A release-latch stops it double-firing, and a
relaxed fist has no pinky out, so it can never toggle control.

SCROLL (when control is ON): hold a pointing pose -- index out, middle & ring
curled -- and FLICK your index up/down. Each flick launches a smooth,
decelerating momentum scroll; flick again mid-glide to add momentum.

ENTER (when control is ON): hold a THUMBS UP -- thumb pointing up out of a
closed fist -- for ~0.4s to press the Enter key. Same release-latch as horns, so
one press per thumbs-up; drop the thumb and raise it again to press twice.

A live window (default on; --headless to disable) shows what the detector sees.
macOS needs Camera + Accessibility permission (see README).

Usage:
    python smalltowngirl.py            # window on
    python smalltowngirl.py --headless # console status only
"""

import argparse
import math
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# --- MediaPipe hand landmark indices -----------------------------------------
WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20


def dist(a, b):
    """Euclidean distance between two normalized landmarks."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


WINDOW_NAME = "SmallTownGirl  |  horns=toggle  point+flick=scroll  thumbs-up=enter  q=quit"


class GestureScroller:
    def __init__(self, args, state=None):
        self.args = args
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

        # Optional live controller (the menu-bar app). When present, run() reads
        # show_preview / invert / flick_speed / running from it each frame and
        # writes the current `armed` state back for the menu-bar icon.
        self.state = state
        self._window_open = False

        # -- control state ----------------------------------------------------
        self.armed = False              # is scroll-control currently ON?
        self.horns_since = None         # when the horns pose started (dwell timer)
        self.horns_consumed = False     # latch: one toggle per horns "show"
        self.last_toggle = 0.0          # cooldown between toggles

        # -- enter-key state (thumbs up) --------------------------------------
        self.thumb_since = None         # when the thumbs-up pose started
        self.thumb_consumed = False     # latch: one Enter per thumbs-up "show"
        self.last_enter = 0.0           # cooldown between Enter presses

        # -- scroll / momentum state ------------------------------------------
        self.prev_y = None              # index-tip y last frame (flick speed)
        self.prev_t = None
        self.prev_loop_t = None         # per-frame dt for the momentum step
        self.velocity = 0.0             # scroll velocity (units/sec, + = up)
        self.scroll_accum = 0.0         # fractional scroll carried between frames
        self.last_status = None

        # -- tunables ---------------------------------------------------------
        self.horns_dwell = 0.5          # s to hold horns before it toggles
        self.toggle_cooldown = 0.4      # s min between toggles (release-latch is the main guard)
        self.thumb_dwell = 0.4          # s to hold thumbs up before Enter fires
        self.enter_cooldown = 0.3       # s min between presses (release-latch is the main guard)
        self.settle_delay = 0.25        # s to ignore flicks right after arming
        self.flick_min_speed = args.flick_speed
        self.flick_gain = args.flick_gain
        self.momentum_tau = args.glide
        self.max_velocity = args.max_speed
        self.velocity_floor = 0.5
        self.invert = args.invert

        # -- live telemetry for the HUD --------------------------------------
        self.show_window = not args.headless
        self.tele_horns = False
        self.tele_horns_progress = 0.0
        self.tele_pointing = False
        self.tele_velocity = 0.0
        self.tele_thumb = False
        self.tele_thumb_progress = 0.0
        self.recent_toggle = None       # (armed_bool, timestamp) for the flash
        self.recent_enter = None        # timestamp of the last Enter, for the flash

    # -- live controller sync -------------------------------------------------
    def _sync_from_state(self):
        """Pull live settings from the menu-bar controller (if any)."""
        s = self.state
        if s is None:
            return
        self.show_window = s.show_preview
        self.invert = s.invert
        self.flick_min_speed = s.flick_speed

    # -- pose helpers ---------------------------------------------------------
    @staticmethod
    def _extended(lm, tip, pip):
        """A finger is extended when its tip is farther from the wrist than its PIP."""
        return dist(lm[tip], lm[WRIST]) > dist(lm[pip], lm[WRIST])

    def _fingers(self, lm):
        """(index, middle, ring, pinky) extension booleans. Thumb is ignored
        (its extended/curled state is the least reliable to detect)."""
        return (self._extended(lm, INDEX_TIP, INDEX_PIP),
                self._extended(lm, MIDDLE_TIP, MIDDLE_PIP),
                self._extended(lm, RING_TIP, RING_PIP),
                self._extended(lm, PINKY_TIP, PINKY_PIP))

    def is_horns(self, lm):
        idx, mid, ring, pinky = self._fingers(lm)
        return idx and pinky and not mid and not ring

    def is_thumbs_up(self, lm):
        """Thumb sticking up out of a closed fist.

        The wrist-distance test used for the other fingers is unreliable for the
        thumb, so this checks two things directly: the thumb reaches well past
        its own base knuckle, and the tip sits clearly above that knuckle on
        screen. Distances are scaled by the wrist->index-MCP span so the test
        holds at any distance from the camera.
        """
        idx, mid, ring, pinky = self._fingers(lm)
        if idx or mid or ring or pinky:
            return False
        scale = dist(lm[WRIST], lm[INDEX_MCP])
        if scale <= 0:
            return False
        if dist(lm[THUMB_TIP], lm[THUMB_MCP]) < 0.7 * scale:
            return False                     # thumb tucked into the fist
        return (lm[THUMB_MCP].y - lm[THUMB_TIP].y) > 0.55 * scale

    def is_pointing(self, lm):
        # Index extended, middle & ring curled (pinky ignored) — the original
        # flick gate. Horns hands are excluded separately when picking the
        # scroll hand, so the pinky doesn't need to matter here.
        idx, mid, ring, pinky = self._fingers(lm)
        return idx and not mid and not ring

    def status(self, msg):
        if msg != self.last_status:
            print(msg)
            self.last_status = msg

    # -- control toggle (horns dwell on either hand) --------------------------
    def update_control(self, hands, now):
        horns = any(self.is_horns(lm) for lm in hands)
        self.tele_horns = horns

        if not horns:
            self.horns_since = None
            self.horns_consumed = False
            self.tele_horns_progress = 0.0
            return

        if self.horns_since is None:
            self.horns_since = now
        progress = (now - self.horns_since) / self.horns_dwell
        self.tele_horns_progress = min(progress, 1.0)

        if (progress >= 1.0 and not self.horns_consumed
                and (now - self.last_toggle) > self.toggle_cooldown):
            self.armed = not self.armed
            self.last_toggle = now
            self.horns_consumed = True          # require release before next toggle
            self.recent_toggle = (self.armed, now)
            self.velocity = 0.0                  # deterministic stop on OFF
            self.scroll_accum = 0.0
            print(f"[toggle] scroll control {'ON' if self.armed else 'OFF'}")

    # -- enter key (thumbs-up dwell on either hand) ---------------------------
    def update_enter(self, hands, now):
        """Hold a thumbs up for `thumb_dwell` seconds (0.4s) to press Enter.

        Only active while control is ON, so a casual thumbs up can't type into
        whatever happens to be focused.
        """
        thumb = self.armed and any(self.is_thumbs_up(lm) for lm in hands)
        self.tele_thumb = thumb

        if not thumb:
            self.thumb_since = None
            self.thumb_consumed = False
            self.tele_thumb_progress = 0.0
            return

        if self.thumb_since is None:
            self.thumb_since = now
        progress = (now - self.thumb_since) / self.thumb_dwell
        self.tele_thumb_progress = min(progress, 1.0)

        if (progress >= 1.0 and not self.thumb_consumed
                and (now - self.last_enter) > self.enter_cooldown):
            self.keyboard.press(Key.enter)
            self.keyboard.release(Key.enter)
            self.last_enter = now
            self.thumb_consumed = True       # require release before next press
            self.recent_enter = now
            print("[enter] pressed")

    # -- flick injection on the scrolling hand --------------------------------
    def update_scroll(self, lm, now):
        if (now - self.last_toggle) < self.settle_delay:
            self.prev_y, self.prev_t = lm[INDEX_TIP].y, now
            return
        y = lm[INDEX_TIP].y
        if self.prev_y is not None and self.prev_t is not None:
            dt = now - self.prev_t
            if dt > 0:
                dy = y - self.prev_y                 # y grows downward
                if abs(dy) / dt > self.flick_min_speed:
                    self.velocity += -dy * self.flick_gain   # up (dy<0) => +vel
                    self.velocity = max(-self.max_velocity,
                                        min(self.max_velocity, self.velocity))
        self.prev_y, self.prev_t = y, now

    # -- momentum integration (every frame, hand or not) ----------------------
    def step_momentum(self, now):
        dt = (now - self.prev_loop_t) if self.prev_loop_t is not None else 0.0
        self.prev_loop_t = now

        if not self.armed or dt <= 0:
            self.velocity = 0.0
            self.scroll_accum = 0.0
            self.tele_velocity = 0.0
            return

        self.velocity *= math.exp(-dt / self.momentum_tau)   # friction glide
        if abs(self.velocity) < self.velocity_floor:
            self.velocity = 0.0

        self.scroll_accum += self.velocity * dt
        steps = int(self.scroll_accum)
        if steps != 0:
            self.scroll_accum -= steps
            self.mouse.scroll(0, -steps if self.invert else steps)
        self.tele_velocity = self.velocity

    # -- window drawing -------------------------------------------------------
    @staticmethod
    def draw_hand(frame, lm, color):
        h, w = frame.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for c in HAND_CONNECTIONS:
            cv2.line(frame, pts[c.start], pts[c.end], color, 2)
        for i, (x, y) in enumerate(pts):
            r = 8 if i in (THUMB_TIP, INDEX_TIP) else 4
            cv2.circle(frame, (x, y), r, (0, 0, 255), -1)

    def draw_hud(self, frame, now):
        h, w = frame.shape[:2]
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        bx, bw = 155, 155

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, 202), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # CONTROL
        c = (80, 220, 80) if self.armed else (80, 80, 235)
        cv2.putText(frame, f"CONTROL: {'ON' if self.armed else 'OFF'}", (24, 42),
                    FONT, 0.8, c, 2)

        # HORNS dwell progress toward a toggle
        hc = (0, 220, 255) if self.tele_horns else (150, 150, 150)
        cv2.putText(frame, f"HORNS:{'YES' if self.tele_horns else 'no'}", (24, 76),
                    FONT, 0.55, hc, 2)
        by = 64
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 12), (90, 90, 90), 1)
        cv2.rectangle(frame, (bx, by), (bx + int(self.tele_horns_progress * bw), by + 12), hc, -1)

        # THUMB dwell progress toward an Enter press
        tc = (235, 90, 220) if self.tele_thumb else (150, 150, 150)
        cv2.putText(frame, f"THUMB:{'YES' if self.tele_thumb else 'no'}", (24, 108),
                    FONT, 0.55, tc, 2)
        ty = 96
        cv2.rectangle(frame, (bx, ty), (bx + bw, ty + 12), (90, 90, 90), 1)
        cv2.rectangle(frame, (bx, ty), (bx + int(self.tele_thumb_progress * bw), ty + 12), tc, -1)

        # SCROLL-hand pointing gate
        gc = (80, 220, 80) if self.tele_pointing else (120, 120, 120)
        cv2.putText(frame, f"POINT:{'YES' if self.tele_pointing else 'no'}", (24, 140),
                    FONT, 0.55, gc, 2)

        # momentum bar (left = up, right = down)
        cv2.putText(frame, "SCROLL:", (24, 177), FONT, 0.55, (180, 180, 180), 2)
        my = 170
        cv2.rectangle(frame, (bx, my - 10), (bx + bw, my + 10), (60, 60, 60), 1)
        cx = bx + bw // 2
        cv2.line(frame, (cx, my - 12), (cx, my + 12), (110, 110, 110), 1)
        frac = max(-1.0, min(1.0, self.tele_velocity / self.max_velocity))
        mag = int(abs(frac) * (bw // 2))
        mc = (0, 220, 255) if abs(self.tele_velocity) >= self.velocity_floor else (120, 120, 120)
        if frac >= 0:
            cv2.rectangle(frame, (cx - mag, my - 8), (cx, my + 8), mc, -1)
        else:
            cv2.rectangle(frame, (cx, my - 8), (cx + mag, my + 8), mc, -1)

        # toggle / enter flash (a toggle wins if both are fresh)
        if self.recent_toggle and (now - self.recent_toggle[1]) < 0.9:
            on = self.recent_toggle[0]
            txt = "CONTROL ON" if on else "CONTROL OFF"
            col = (80, 230, 80) if on else (80, 80, 235)
        elif self.recent_enter and (now - self.recent_enter) < 0.9:
            txt, col = "ENTER", (235, 90, 220)
        else:
            txt = None
        if txt:
            (tw, _), _ = cv2.getTextSize(txt, FONT, 1.2, 3)
            cv2.putText(frame, txt, ((w - tw) // 2, h - 34), FONT, 1.2, col, 3)

    # -- main loop ------------------------------------------------------------
    def run(self):
        if not os.path.exists(self.args.model):
            print(f"ERROR: model file not found: {self.args.model}", file=sys.stderr)
            sys.exit(1)

        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print(f"ERROR: could not open camera index {self.args.camera}.", file=sys.stderr)
            sys.exit(1)

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=self.args.model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        start = time.time()

        print("SmallTownGirl ready. Hold HORNS (index+pinky) ~1s to toggle. Point + flick to scroll. "
              "Hold THUMBS UP ~1s for Enter. q/Ctrl+C to quit.")
        try:
            while True:
                self._sync_from_state()
                if self.state is not None and not self.state.running:
                    break
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)  # mirror for intuitive control
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now = time.time()
                result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))

                if result.hand_landmarks:
                    hands = result.hand_landmarks
                    self.update_control(hands, now)
                    self.update_enter(hands, now)

                    scroll_lm = next((lm for lm in hands
                                      if self.is_pointing(lm) and not self.is_horns(lm)), None)
                    self.tele_pointing = scroll_lm is not None
                    if self.armed and scroll_lm is not None:
                        self.update_scroll(scroll_lm, now)
                    else:
                        self.prev_y = None

                    if self.show_window:
                        for lm in hands:
                            if self.is_horns(lm):
                                col = (0, 220, 255)     # horns -> cyan
                            elif self.is_thumbs_up(lm):
                                col = (235, 90, 220)     # thumbs up -> magenta
                            elif self.is_pointing(lm):
                                col = (0, 200, 0)        # pointing -> green
                            else:
                                col = (140, 140, 140)    # other -> grey
                            self.draw_hand(frame, lm, col)
                    self.status("control: " + ("ON " if self.armed else "OFF")
                                + ("  (point + flick to scroll, thumbs up for enter)" if self.armed
                                   else "  (horns to arm)"))
                else:
                    self.prev_y = None
                    self.horns_since = None
                    self.horns_consumed = False
                    self.thumb_since = None
                    self.thumb_consumed = False
                    self.tele_horns = False
                    self.tele_horns_progress = 0.0
                    self.tele_thumb = False
                    self.tele_thumb_progress = 0.0
                    self.tele_pointing = False

                # Momentum advances every frame so the glide keeps going even
                # after the hand leaves the frame.
                self.step_momentum(now)

                # Publish current control state for the menu-bar icon.
                if self.state is not None:
                    self.state.armed = self.armed

                if self.show_window:
                    self.draw_hud(frame, now)
                    cv2.imshow(WINDOW_NAME, frame)
                    self._window_open = True
                    # In the menu-bar app, 'q' just closes the preview; only the
                    # standalone CLI quits on 'q'.
                    if cv2.waitKey(1) & 0xFF == ord("q") and self.state is None:
                        break
                elif self._window_open:
                    cv2.destroyWindow(WINDOW_NAME)
                    cv2.waitKey(1)
                    self._window_open = False
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            cap.release()
            landmarker.close()
            if self._window_open:
                cv2.destroyAllWindows()
                cv2.waitKey(1)


def parse_args():
    p = argparse.ArgumentParser(description="SmallTownGirl — Hand Gesture based PC control "
                                            "(horns toggle + momentum flick + thumbs-up enter).")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--model", default=os.environ.get("GESTURE_SCROLL_MODEL", MODEL_PATH),
                   help="path to the hand_landmarker.task model file")
    p.add_argument("--headless", action="store_true",
                   help="run without the detection window (console status only)")
    p.add_argument("--invert", action="store_false", help="invert scroll direction")
    p.add_argument("--flick-speed", type=float, default=0.9, dest="flick_speed",
                   help="min finger speed (norm units/sec) to register a flick; higher = fewer false triggers")
    p.add_argument("--flick-gain", type=float, default=400.0, dest="flick_gain",
                   help="how much scroll velocity a flick injects")
    p.add_argument("--glide", type=float, default=0.45, dest="glide",
                   help="glide time constant in seconds; larger = longer, smoother coast")
    p.add_argument("--max-speed", type=float, default=120.0, dest="max_speed",
                   help="cap on scroll velocity (units/sec)")
    return p.parse_args()


if __name__ == "__main__":
    GestureScroller(parse_args()).run()
