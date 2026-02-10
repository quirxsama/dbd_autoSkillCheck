import os
import sys
from time import time, sleep
import gradio as gr

from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE
from dbd.utils.humanizer import humanized_press
from dbd.utils.monitoring_mss import Monitoring_mss

# Optional: BetterCam (Windows only)
try:
    from dbd.utils.monitoring_bettercam import Monitoring_bettercam
    bettercam_ok = True
    print("Info: BetterCam feature available (Windows).")
except ImportError:
    bettercam_ok = False

# Optional: v4l2loopback / OBS VirtualCam (Linux only)
try:
    from dbd.utils.monitoring_v4l2 import Monitoring_v4l2, V4L2_AVAILABLE
    v4l2_ok = V4L2_AVAILABLE
    if v4l2_ok:
        print("Info: v4l2 (OBS VirtualCam) feature available (Linux).")
except ImportError:
    v4l2_ok = False
    V4L2_AVAILABLE = False


# Detect platform
def get_platform_info():
    """Detect the current platform and display environment."""
    info = {"os": sys.platform, "display": "unknown"}
    
    if sys.platform == "win32":
        info["display"] = "Windows"
    elif sys.platform == "darwin":
        info["display"] = "macOS"
    else:
        # Linux - check for Wayland or X11
        session_type = os.environ.get('XDG_SESSION_TYPE', '')
        wayland_display = os.environ.get('WAYLAND_DISPLAY', '')
        
        if session_type == 'wayland' or wayland_display:
            info["display"] = "Wayland"
        else:
            info["display"] = "X11"
    
    return info

platform_info = get_platform_info()
print(f"Info: Platform detected: {platform_info['display']}")


ai_model = None
devices = ["CPU (default)", "GPU"]

def cleanup():
    global ai_model
    if ai_model is not None:
        del ai_model
        ai_model = None
    return 0.


# FPS Presets
FPS_PRESETS = {
    "Custom": None,
    "60 FPS (locked)": -250,
    "90 FPS (locked)": -125,
    "120 FPS": 0,
}


def apply_fps_preset(preset_name):
    """Apply ante-frontier delay based on FPS preset."""
    if preset_name in FPS_PRESETS and FPS_PRESETS[preset_name] is not None:
        return gr.update(value=FPS_PRESETS[preset_name])
    return gr.skip()


def monitor(ai_model_path, device, monitoring_str, monitor_id, hit_ante, nb_cpu_threads, use_hesitation):
    if ai_model_path is None or not os.path.exists(ai_model_path):
        raise gr.Error("Invalid AI model file", duration=0)

    if device is None:
        raise gr.Error("Invalid device option")

    if monitor_id is None:
        raise gr.Error("Invalid monitor option")

    use_gpu = (device == devices[1])

    if monitoring_str == "v4l2 (OBS VirtualCam)" and v4l2_ok:
        monitoring = Monitoring_v4l2(device_id=monitor_id, crop_size=224)
    elif monitoring_str == "bettercam" and bettercam_ok:
        monitoring = Monitoring_bettercam(monitor_id=monitor_id, crop_size=224, target_fps=240)
    else:
        monitoring = Monitoring_mss(monitor_id=monitor_id, crop_size=224)

    try:
        global ai_model
        model_instance = AI_model(ai_model_path, use_gpu, nb_cpu_threads, monitoring)
        ai_model = model_instance
        execution_provider = model_instance.check_provider()
    except Exception as e:
        raise gr.Error("Error when loading AI model: {}".format(e), duration=0)

    if execution_provider == "CUDAExecutionProvider":
        gr.Info("Running AI model on GPU (success, CUDA)")
    elif execution_provider == "DmlExecutionProvider":
        gr.Info("Running AI model on GPU (success, DirectML)")
    elif execution_provider == "TensorRT":
        gr.Info("Running AI model on GPU (success, TensorRT)")
    else:
        gr.Info(f"Running AI model on CPU (success, {nb_cpu_threads} threads)")
        if use_gpu:
            Warning("Could not run AI model on GPU device. Check python console logs to debug.")

    # Variables
    t0 = time()
    nb_frames = 0

    try:
        while True:
            if model_instance is None:
                break
                
            frame_np = model_instance.grab_screenshot()
            nb_frames += 1

            pred, desc, probs, should_hit = model_instance.predict(frame_np)

            if should_hit:
                # ante-frontier hit delay
                if pred == 2 and hit_ante > 0:
                    sleep(hit_ante * 0.001)

                # Humanized key press
                cooldown = humanized_press(
                    SPACE, PressKey, ReleaseKey, use_hesitation=use_hesitation
                )

                yield gr.skip(), frame_np, probs

                sleep(cooldown)  # humanized cooldown
                t0 = time()
                nb_frames = 0
                continue

            # Compute fps
            t_diff = time() - t0
            if t_diff > 1.0:
                fps = round(nb_frames / t_diff, 1)
                yield fps, gr.skip(), gr.skip()

                t0 = time()
                nb_frames = 0

    except Exception as e:
        print(f"Monitor loop error: {e}")
        pass
    finally:
        print("Monitoring stopped.")


if __name__ == "__main__":
    models_folder = "models"

    fps_info = "Number of frames per second the AI model analyses the monitored frame."
    cpu_choices = [("Low", 2), ("Normal", 4), ("High", 6), ("CPU BBQ Mode", 8)]

    # Find available AI models
    model_files = [(f, f'{models_folder}/{f}') for f in os.listdir(f"{models_folder}/") if f.endswith(".onnx") or f.endswith(".trt")]
    if len(model_files) == 0:
        raise gr.Error(f"No AI model found in {models_folder}/", duration=0)

    # Build monitoring choices based on platform
    monitoring_choices = ["mss"]
    monitoring_tooltips = {
        "mss": "Cross-platform screen capture. Works on Windows and Linux (X11). Does NOT work on Wayland.",
        "bettercam": "Windows-only high-performance screen capture. Recommended for Windows users.",
        "v4l2 (OBS VirtualCam)": "Linux screen capture via OBS Virtual Camera. Works on both X11 and Wayland. Requires OBS with 'Start Virtual Camera' enabled.",
    }
    
    if bettercam_ok:
        monitoring_choices.insert(0, "bettercam")
    if v4l2_ok:
        monitoring_choices.append("v4l2 (OBS VirtualCam)")

    # Auto-select best default monitoring method
    if platform_info["display"] == "Windows" and bettercam_ok:
        default_monitoring = "bettercam"
    elif platform_info["display"] == "Wayland" and v4l2_ok:
        default_monitoring = "v4l2 (OBS VirtualCam)"
    else:
        default_monitoring = "mss"

    # Build monitoring info text
    monitoring_info_lines = []
    for method in monitoring_choices:
        tip = monitoring_tooltips.get(method, "")
        monitoring_info_lines.append(f"• **{method}**: {tip}")
    monitoring_info_text = "\n".join(monitoring_info_lines)

    def switch_monitoring_cb(monitoring_str):
        if monitoring_str == "v4l2 (OBS VirtualCam)" and v4l2_ok:
            monitor_choices = Monitoring_v4l2.get_monitors_info()
        elif monitoring_str == "bettercam" and bettercam_ok:
            monitor_choices = Monitoring_bettercam.get_monitors_info()
        else:
            monitor_choices = Monitoring_mss.get_monitors_info()

        return gr.update(choices=monitor_choices, value=None), None

    # Monitor selection
    if default_monitoring == "v4l2 (OBS VirtualCam)" and v4l2_ok:
        monitor_choices = Monitoring_v4l2.get_monitors_info()
    elif default_monitoring == "bettercam" and bettercam_ok:
        monitor_choices = Monitoring_bettercam.get_monitors_info()
    else:
        monitor_choices = Monitoring_mss.get_monitors_info()

    def switch_monitor_cb(monitoring_str, monitor_id):
        try:
            if monitoring_str == "v4l2 (OBS VirtualCam)" and v4l2_ok:
                with Monitoring_v4l2(monitor_id, crop_size=520) as mon:
                    return mon.get_frame_np()
            elif monitoring_str == "bettercam" and bettercam_ok:
                with Monitoring_bettercam(monitor_id, crop_size=520) as mon:
                    return mon.get_frame_np()
            else:
                with Monitoring_mss(monitor_id, crop_size=520) as mon:
                    return mon.get_frame_np()
        except Exception as e:
            print(f"Monitor preview error: {e}")
            return None

    # --- Web UI ---
    with (gr.Blocks(title="DBD Auto Skill Check") as webui):
        gr.Markdown("# <center>🎮 DBD Auto Skill Check</center>", elem_id="title")
        gr.Markdown(
            f"<center>"
            f"[GitHub](https://github.com/Manuteaa/dbd_autoSkillCheck) • "
            f"Platform: **{platform_info['display']}**"
            f"</center>"
        )

        with gr.Row():
            with gr.Column(variant="panel"):
                # AI Settings
                with gr.Column(variant="panel"):
                    gr.Markdown("### ⚙️ AI Inference Settings")
                    ai_model_path = gr.Dropdown(
                        choices=model_files,
                        value=model_files[0][1],
                        label="AI Model (ONNX or TensorRT Engine)",
                        info="Select the trained AI model to use for skill check detection."
                    )
                    device = gr.Radio(
                        choices=devices,
                        value=devices[0],
                        label="Device",
                        info="CPU works for most users. GPU requires CUDA/DirectML setup (see README)."
                    )
                    with gr.Row():
                        monitoring_str = gr.Dropdown(
                            choices=monitoring_choices,
                            value=default_monitoring,
                            label="Screen Capture Method",
                            info=f"Auto-detected: {default_monitoring}. Hover for details."
                        )
                        monitor_id = gr.Dropdown(
                            choices=monitor_choices,
                            value=monitor_choices[0][1] if monitor_choices else 0,
                            label="Monitor / Source",
                            info="Select the monitor or capture source where you play the game."
                        )
                    
                    # Show monitoring method descriptions
                    with gr.Accordion("📖 Screen Capture Methods Info", open=False):
                        gr.Markdown(monitoring_info_text)
                        if platform_info["display"] == "Wayland":
                            gr.Markdown(
                                "> ⚠️ **Wayland detected:** Use `v4l2 (OBS VirtualCam)` for best results. "
                                "`mss` does **not** work on Wayland. "
                                "Open OBS, add your game window, then click **Start Virtual Camera**."
                            )

                # Feature Options
                with gr.Column(variant="panel"):
                    gr.Markdown("### 🎯 Feature Options")
                    
                    fps_preset = gr.Radio(
                        choices=list(FPS_PRESETS.keys()),
                        value="Custom",
                        label="FPS Preset",
                        info="Quick presets for ante-frontier delay based on your game FPS. "
                             "60 FPS → -250ms, 90 FPS → -125ms, 120+ FPS → 0ms."
                    )
                    
                    hit_ante = gr.Slider(
                        minimum=-300, maximum=50, step=5, value=0,
                        label="Ante-frontier hit delay (ms)",
                        info="Negative = hit earlier (compensate for input lag at lower FPS). "
                             "Positive = hit later. Adjust based on your game FPS and ping."
                    )
                    
                    cpu_stress = gr.Radio(
                        label="CPU Workload",
                        choices=cpu_choices,
                        value=cpu_choices[1][1],
                        info="Increase to improve AI FPS, decrease to reduce CPU usage. "
                             "Adjust based on your hardware."
                    )
                    
                    use_hesitation = gr.Checkbox(
                        label="Human-like Hesitation (Recommended)",
                        value=True,
                        info="Adds random micro-delays (~7% chance). Increases safety against anti-cheat, "
                             "but creates a small chance of hitting 'Good' instead of 'Great'."
                    )

                # Controls
                with gr.Column():
                    run_button = gr.Button("▶ RUN", variant="primary", size="lg")
                    stop_button = gr.Button("⏹ STOP", variant="stop", size="lg")

            # Right panel - Output
            with gr.Column(variant="panel"):
                gr.Markdown("### 📊 Live Monitoring")
                fps = gr.Number(label="AI Model FPS", info=fps_info, interactive=False)
                image_visu = gr.Image(label="Last hit skill check frame", height=224, interactive=False)
                probs = gr.Label(label="Skill Check AI Recognition")

        # Event handlers
        monitoring = run_button.click(
            fn=monitor, 
            inputs=[ai_model_path, device, monitoring_str, monitor_id, hit_ante, cpu_stress, use_hesitation],
            outputs=[fps, image_visu, probs]
        )

        stop_button.click(fn=cleanup, inputs=None, outputs=fps)
        fps_preset.change(fn=apply_fps_preset, inputs=[fps_preset], outputs=[hit_ante])
        monitoring_str.blur(fn=switch_monitoring_cb, inputs=[monitoring_str], outputs=[monitor_id, image_visu])
        monitor_id.blur(fn=switch_monitor_cb, inputs=[monitoring_str, monitor_id], outputs=image_visu)

    try:
        webui.launch(theme=gr.themes.Soft())
    except:
        print("User stopped the web UI. Please wait to cleanup resources...")
    finally:
        cleanup()
