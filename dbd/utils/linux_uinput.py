# linux_uinput.py
# Advanced Linux Kernel-Level Input Injection using evdev/uinput.
#
# This module creates a virtual input device at the kernel level.
# To Anti-Cheat systems, input from this device appears identical to
# physical hardware interrupts, avoiding the LLKHF_INJECTED flag used by
# user-space tools like xdotool or pynput.
#
# FEATURE: Polymorphic Device Spoofing
# Randomizes Vendor/Product IDs and device names on initialization to
# prevent static signature blacklisting.

import time
import random
import sys
import os
from typing import Optional

# Graceful import handling
try:
    import evdev
    from evdev import UInput, ecodes as e, AbsInfo
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False

# ── Hardware Identity Database ───────────────────────────────────────────────
# Real hardware IDs to mimic legitimate gaming peripherals.
# Format: (VendorID, ProductID, "Device Name")

HARDWARE_DB = [
    (0x046d, 0xc339, "Logitech G Pro Gaming Keyboard"),
    (0x046d, 0xc33f, "Logitech G815 RGB Mechanical Gaming Keyboard"),
    (0x1532, 0x0203, "Razer BlackWidow Chroma"),
    (0x1532, 0x021e, "Razer Ornata Chroma"),
    (0x1b1c, 0x1b13, "Corsair K70 RGB MK.2 Mechanical Gaming Keyboard"),
    (0x1b1c, 0x1b2d, "Corsair K95 RGB PLATINUM XT"),
    (0x0951, 0x16a4, "HyperX Alloy FPS Pro Mechanical Gaming Keyboard"),
    (0x1038, 0x1610, "SteelSeries Apex Pro"),
    (0x045e, 0x00db, "Microsoft Natural Ergonomic Keyboard 4000"),
    (0x045e, 0x07f8, "Microsoft Sidewinder X4 Keyboard"),
]

class LinuxVirtualController:
    def __init__(self):
        self.uinput: Optional['UInput'] = None
        self.device_name = "Unknown"
        
        if not EVDEV_AVAILABLE:
            print("[Warning] 'evdev' library not found. Falling back to pynput.")
            return

        try:
            self._create_device()
        except PermissionError:
            print("[Error] Permission denied accessing /dev/uinput.")
            print("  -> Run 'sudo chmod +0666 /dev/uinput' or add udev rules.")
            print("  -> Falling back to standard pynput (less safe).")
            self.uinput = None
        except Exception as e:
            print(f"[Error] Failed to create virtual device: {e}")
            self.uinput = None

    def _create_device(self):
        # 1. Select a random persona
        vid, pid, name = random.choice(HARDWARE_DB)
        
        # 2. Add slight randomness to name to avoid exact string matching blocks
        # (e.g., "Logitech G Pro" -> "Logitech G Pro (USB)")
        suffixes = ["", " (USB)", " Gaming Device", " Interface", " v2"]
        self.device_name = name + random.choice(suffixes)
        
        # 3. Define Capabilities
        # IMPORTANT: We must declare support for MANY keys, not just Space.
        # A keyboard that only has a Spacebar is extremely suspicious.
        cap = {
            e.EV_KEY: [
                e.KEY_SPACE, e.KEY_ENTER, e.KEY_ESC,
                e.KEY_A, e.KEY_B, e.KEY_C, e.KEY_D, e.KEY_E, e.KEY_F, e.KEY_G,
                e.KEY_LEFTSHIFT, e.KEY_LEFTCTRL, e.KEY_LEFTALT,
                e.KEY_1, e.KEY_2, e.KEY_3, e.KEY_4, e.KEY_5
            ],
            # Add basic relative axis support (mouse-like) to look like a precise composite device
            # e.EV_REL: [e.REL_X, e.REL_Y] 
        }

        # 4. Initialize UInput Device
        self.uinput = UInput(
            events=cap,
            name=self.device_name,
            vendor=vid,
            product=pid,
            version=0x111  # version 1.1.1
        )
        print(f"[Core] Virtual Kernel Device Initialized: {self.device_name} ({hex(vid)}:{hex(pid)})")

    def press(self, key_code):
        """Send specific key DOWN event."""
        if self.uinput:
            # Map common internal key codes to linux ecodes if needed, 
            # but for now we assume key_code corresponds to evdev constants usually.
            # Space mapping:
            target = e.KEY_SPACE if key_code == 'space' else key_code
            
            self.uinput.write(e.EV_KEY, target, 1) # 1 = Down
            self.uinput.syn()

    def release(self, key_code):
        """Send specific key UP event."""
        if self.uinput:
            target = e.KEY_SPACE if key_code == 'space' else key_code
            
            self.uinput.write(e.EV_KEY, target, 0) # 0 = Up
            self.uinput.syn()
            
    def is_active(self):
        return self.uinput is not None
    
    def close(self):
        if self.uinput:
            self.uinput.close()

# Singleton instance
_vcontroller = None

def get_controller():
    global _vcontroller
    if _vcontroller is None:
        _vcontroller = LinuxVirtualController()
    return _vcontroller
