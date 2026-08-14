"""
SmallTownGirl — macOS menu-bar app.

A lightweight status-bar agent (no dock icon) that runs the gesture-detection
engine in a background thread and exposes preferences:
  * Active               — start/stop detection (frees the camera when off)
  * Show Camera Preview  — headless (default) vs. a live detection window
  * Invert Scroll
  * Sensitivity          — Low / Medium / High (flick threshold)
  * Start at Login

Preferences persist to ~/Library/Application Support/SmallTownGirl/prefs.json.
The menu-bar icon reflects state: ⏸ inactive · 🖐 armed-off · 🤘 control ON.
"""

import argparse
import json
import os
import threading

import rumps

from smalltowngirl import GestureScroller

APP_NAME = "SmallTownGirl"
BUNDLE_ID = "com.piyushkhanna.smalltowngirl"
DATA_DIR = os.path.expanduser("~/Library/Application Support/SmallTownGirl")
PREFS_PATH = os.path.join(DATA_DIR, "prefs.json")
MODEL = os.environ.get("GESTURE_SCROLL_MODEL", os.path.join(DATA_DIR, "hand_landmarker.task"))

SENSITIVITY = {"Low": 1.3, "Medium": 0.9, "High": 0.6}   # label -> flick_speed
DEFAULT_PREFS = {"active": True, "show_preview": False, "invert": False, "sensitivity": "Medium"}


class State:
    """Shared object the detection thread reads each frame and writes `armed` to."""
    def __init__(self, prefs):
        self.running = False
        self.show_preview = prefs["show_preview"]
        self.invert = prefs["invert"]
        self.flick_speed = SENSITIVITY.get(prefs["sensitivity"], 0.9)
        self.armed = False


def load_prefs():
    try:
        with open(PREFS_PATH) as f:
            return {**DEFAULT_PREFS, **json.load(f)}
    except Exception:
        return dict(DEFAULT_PREFS)


def save_prefs(prefs):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = PREFS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prefs, f, indent=2)
    os.replace(tmp, PREFS_PATH)


def _engine_args():
    # Live-tunable fields (invert / flick_speed / show_preview) come from State;
    # these are the fixed defaults for everything else.
    return argparse.Namespace(
        camera=0, model=MODEL, headless=True, invert=False,
        flick_speed=0.9, flick_gain=320.0, glide=0.45, max_speed=120.0,
    )


def login_plist_path():
    return os.path.expanduser(f"~/Library/LaunchAgents/{BUNDLE_ID}.plist")


def set_login_item(enable):
    path = login_plist_path()
    if enable:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{BUNDLE_ID}</string>
  <key>ProgramArguments</key>
  <array><string>/usr/bin/open</string><string>/Applications/SmallTownGirl.app</string></array>
  <key>RunAtLoad</key><true/>
</dict></plist>
"""
        with open(path, "w") as f:
            f.write(plist)
    elif os.path.exists(path):
        os.remove(path)


class SmallTownGirlApp(rumps.App):
    def __init__(self):
        super().__init__(APP_NAME, title="🖐", quit_button=None)
        self.prefs = load_prefs()
        self.state = State(self.prefs)
        self.worker = None

        self._mi_active = rumps.MenuItem("Active", callback=self.toggle_active)
        self._mi_preview = rumps.MenuItem("Show Camera Preview", callback=self.toggle_preview)
        self._mi_invert = rumps.MenuItem("Invert Scroll", callback=self.toggle_invert)
        self._mi_login = rumps.MenuItem("Start at Login", callback=self.toggle_login)
        self._sens = {k: rumps.MenuItem(k, callback=self.set_sensitivity) for k in SENSITIVITY}

        self.menu = [
            self._mi_active,
            None,
            self._mi_preview,
            self._mi_invert,
            ("Sensitivity", list(self._sens.values())),
            None,
            self._mi_login,
            None,
            rumps.MenuItem("About", callback=self.about),
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]
        self._refresh_checks()

        # Poll shared state and update the menu-bar glyph on the main thread.
        self._ticker = rumps.Timer(self._tick, 0.3)
        self._ticker.start()

        if self.prefs["active"]:
            self.start_worker()

    # -- worker management ----------------------------------------------------
    def start_worker(self):
        if self.worker and self.worker.is_alive():
            return
        if not os.path.exists(MODEL):
            self._notify("Model missing", "Re-run first-time setup.")
            return
        self.state.running = True
        self.worker = threading.Thread(target=self._run_loop, daemon=True)
        self.worker.start()

    def stop_worker(self):
        self.state.running = False   # loop sees this, returns, releases the camera

    def _run_loop(self):
        try:
            GestureScroller(_engine_args(), state=self.state).run()
        except Exception as exc:  # noqa: BLE001 - surface to the user, keep app alive
            self._notify("Detection stopped", str(exc))
        finally:
            self.state.armed = False

    # -- menu callbacks -------------------------------------------------------
    def toggle_active(self, _):
        self.prefs["active"] = not self.prefs["active"]
        self.start_worker() if self.prefs["active"] else self.stop_worker()
        save_prefs(self.prefs)
        self._refresh_checks()

    def toggle_preview(self, _):
        self.prefs["show_preview"] = self.state.show_preview = not self.prefs["show_preview"]
        save_prefs(self.prefs)
        self._refresh_checks()

    def toggle_invert(self, _):
        self.prefs["invert"] = self.state.invert = not self.prefs["invert"]
        save_prefs(self.prefs)
        self._refresh_checks()

    def set_sensitivity(self, sender):
        self.prefs["sensitivity"] = sender.title
        self.state.flick_speed = SENSITIVITY[sender.title]
        save_prefs(self.prefs)
        self._refresh_checks()

    def toggle_login(self, _):
        set_login_item(not os.path.exists(login_plist_path()))
        self._refresh_checks()

    def about(self, _):
        rumps.alert(
            APP_NAME,
            "Hand-gesture control for your Mac.\n\n"
            "Hold a horns pose (index + pinky) ~1s to toggle control on/off, "
            "then point your index finger and flick up or down to scroll, "
            "or hold a thumbs up ~1s to press Enter.",
        )

    def quit_app(self, _):
        self.stop_worker()
        rumps.quit_application()

    # -- ui refresh -----------------------------------------------------------
    def _tick(self, _):
        self.title = "⏸" if not self.prefs["active"] else ("🤘" if self.state.armed else "🖐")

    def _refresh_checks(self):
        self._mi_active.state = self.prefs["active"]
        self._mi_preview.state = self.prefs["show_preview"]
        self._mi_invert.state = self.prefs["invert"]
        self._mi_login.state = os.path.exists(login_plist_path())
        for k, item in self._sens.items():
            item.state = (self.prefs["sensitivity"] == k)

    def _notify(self, title, msg):
        try:
            rumps.notification(APP_NAME, title, msg)
        except Exception:
            pass


if __name__ == "__main__":
    SmallTownGirlApp().run()
