# SmallTownGirl — Hand Gesture based PC control

Control your Mac with webcam hand gestures — scroll any window, no mouse, no keys.

- **Toggle control:** make a **horns 🤘 pose** with *either* hand — index and
  pinky extended, middle & ring curled — and **hold it ~1 s**. That turns
  scroll-control ON. Hold horns for ~1 s again to turn it OFF. Control persists
  hands-free in between.
- **Scroll (when ON):** hold a **pointing pose** (index out, middle & ring
  curled) and **flick your index up or down**. Each flick launches a smooth,
  decelerating **momentum scroll** — flick again mid-glide to add momentum.

**Why it no longer false-toggles:** control is toggled only by the horns pose,
which requires the pinky extended — a relaxed fist has no pinky out, so it can
never toggle. The 1 s hold plus a release-latch (you must drop horns before it
can fire again) kill accidental and repeat toggles. Scrolling adds a flick-speed
gate so only fast, deliberate motion moves the page.

Built with MediaPipe (hand tracking), OpenCV (camera), and pynput (scroll).

## Install (Homebrew)

```bash
brew install piyush-khanna-qmb/tap/smalltowngirl
smalltowngirl
```

(macOS filesystems are case-insensitive, so `SmallTownGirl` works as the command too.)

The first launch does a one-time setup (~1 min): it builds an isolated Python
environment, installs MediaPipe/OpenCV/pynput, and downloads the hand model into
`~/Library/Application Support/SmallTownGirl`. Subsequent launches are instant.

To update or remove:

```bash
brew upgrade smalltowngirl
brew uninstall smalltowngirl     # then optionally: rm -rf ~/Library/Application\ Support/SmallTownGirl
```

## Run from source (development)

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
curl -sSL -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
./.venv/bin/python smalltowngirl.py
```

## macOS permissions (required)

Grant these to the app you launch it from (Terminal, iTerm, VS Code, etc.) in
**System Settings → Privacy & Security**:

1. **Camera** — so it can see your hand.
2. **Accessibility** — so it can send scroll events to other apps.

After granting, fully quit and reopen that app.

## Run

```bash
# Detection window on by default: camera feed + hand overlay + live HUD
./.venv/bin/python smalltowngirl.py

# No window (console status only)
./.venv/bin/python smalltowngirl.py --headless
```

Quit with `q` in the window (or `Ctrl+C`).

### The detection window

A live HUD shows exactly what the detector sees:

- **CONTROL: ON/OFF** — whether scroll-control is armed (green = on).
- **HORNS: YES/no** + bar — whether a horns pose is detected, and a progress bar
  that fills over the 1 s hold; control toggles the instant it fills.
- **POINT: YES/no** — whether a scrolling hand passes the pointing-pose gate
  (green = yes). Nothing scrolls unless this is YES.
- **SCROLL** momentum bar — fills from center toward *up* or *down* in
  proportion to the current scroll velocity, and shrinks back to center as the
  glide decelerates, so you can watch the momentum bleed off.
- **CONTROL ON / OFF** flash at the bottom each time you toggle.

Each detected hand's skeleton is colored by pose: **cyan** = horns, **green** =
pointing/scroll, grey = neither.

## Tuning

| Flag | Default | Meaning |
|------|---------|---------|
| `--camera N` | 0 | Camera index |
| `--pinch-ratio F` | 0.5 | Lower = must pinch tighter to toggle |
| `--flick-speed F` | 0.9 | Min finger speed to register a flick. **Raise this if you still get false triggers**; lower it if flicks feel unresponsive |
| `--flick-gain F` | 320 | How hard a flick kicks — bigger = scrolls further per flick |
| `--glide F` | 0.45 | Glide time constant (seconds). Bigger = longer, smoother coast |
| `--max-speed F` | 120 | Cap on scroll velocity |
| `--invert` | off | Flip scroll direction |

Watch the HUD while you tune: if stray motions still scroll, raise
`--flick-speed`; if the coast feels too short or abrupt, raise `--glide`.
