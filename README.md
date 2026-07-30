# SmallTownGirl — Hand Gesture based PC control

Control your Mac with webcam hand gestures — scroll any window, no mouse, no keys.

- **Toggle control:** make a **horns 🤘 pose** with *either* hand — index and
  pinky extended, middle & ring curled — and **hold it ~1 s**. That turns
  scroll-control ON. Hold horns for ~1 s again to turn it OFF. Control persists
  hands-free in between.
- **Scroll (when ON):** hold a **pointing pose** (index out, middle & ring
  curled) and **flick your index up or down**. Each flick launches a smooth,
  decelerating **momentum scroll** — flick again mid-glide to add momentum.

Built with MediaPipe (hand tracking), OpenCV (camera), and pynput (scroll).

## Install (menu-bar app)

```bash
brew install --cask piyush-khanna-qmb/tap/smalltowngirl
```

Then launch **SmallTownGirl** from Spotlight or `/Applications`. It runs in your
menu bar (🖐) — there's no dock icon. The first launch does a one-time setup
(~1 min): it builds an isolated Python environment and downloads the hand model
into `~/Library/Application Support/SmallTownGirl`. After that the menu-bar icon
appears and it's instant on every launch.

The app is unsigned, so the **first** time you must clear Gatekeeper — either
right-click `SmallTownGirl.app` in `/Applications` and choose **Open**, or run:

```bash
xattr -dr com.apple.quarantine /Applications/SmallTownGirl.app
```

When macOS prompts, grant **Camera** and **Accessibility**
(System Settings ▸ Privacy & Security) so it can see your hand and send scrolls.

### Menu-bar controls

| Item | What it does |
|------|--------------|
| **Active** | Start/stop detection (frees the camera when off) |
| **Show Camera Preview** | Headless (default) or a live detection window |
| **Invert Scroll** | Flip scroll direction |
| **Sensitivity** | Low / Medium / High (flick threshold) |
| **Start at Login** | Launch automatically when you log in |

Icon states: ⏸ inactive · 🖐 ready · 🤘 control ON.

To update or remove:

```bash
brew upgrade --cask smalltowngirl
brew uninstall --cask smalltowngirl     # `--zap` also removes app data
```

## Run from source (development)

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
curl -sSL -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

./.venv/bin/python menubar.py            # the menu-bar app
./.venv/bin/python smalltowngirl.py      # standalone window (dev/tuning)
./.venv/bin/python smalltowngirl.py --headless
```

To rebuild the `.app` + `.dmg`:  `packaging/build_app.sh 0.2.0`

## macOS permissions (required)

Grant these to **SmallTownGirl** (or, when running from source, to your terminal)
in **System Settings → Privacy & Security**:

1. **Camera** — so it can see your hand.
2. **Accessibility** — so it can send scroll events to other apps.

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
| `--flick-speed F` | 0.9 | Min finger speed to register a flick. **Raise this if you still get false triggers**; lower it if flicks feel unresponsive |
| `--flick-gain F` | 320 | How hard a flick kicks — bigger = scrolls further per flick |
| `--glide F` | 0.45 | Glide time constant (seconds). Bigger = longer, smoother coast |
| `--max-speed F` | 120 | Cap on scroll velocity |
| `--invert` | off | Flip scroll direction |

Watch the HUD while you tune: if stray motions still scroll, raise
`--flick-speed`; if the coast feels too short or abrupt, raise `--glide`.
