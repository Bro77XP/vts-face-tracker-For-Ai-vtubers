# VTube Studio Face Tracker

A Python client for recording, playing back, and processing face tracking data from VTube Studio via its WebSocket API. Includes an LSTM/TCN neural network that learns human motion patterns and generates smooth VTuber animations.

do py -3.12 -m venv venv for amd support for neuro network training for amd gpu.

## Setup

1. Enable **"Allow Plugin API access"** in VTube Studio settings
2. Install dependencies:
```
pip install -r requirements.txt
```

For GPU batch processing (AMD):
```
pip install pyopencl numpy
```

For neural network GPU training (AMD, requires Python 3.12):
```
py -3.12 -m venv venv312
venv312\Scripts\pip install torch-directml pyopencl numpy websockets rich
```

## Files

| File | Description |
|---|---|
| `vts_face_tracker.py` | Main tracker (record, playback, idle export, convert) |
| `vts_face_tracker_amd_gpu.py` | AMD GPU copy (identical) |
| `neural_tracker.py` | Neural network trainer/generator (LSTM, CPU) |
| `neural_tracker_gpu.py` | Neural network trainer (TCN, AMD GPU via DirectML) |
| `gpu_process.py` | Standalone GPU batch processing tool |
| `vts_inspector.py` | VTS parameter inspector utility |

## Usage

**Interactive live display** (default):
```
python vts_face_tracker.py
```

**Show VTS info**:
```
python vts_face_tracker.py --info
```

**List all tracking parameters**:
```
python vts_face_tracker.py --list
```

**Monitor parameters over time** (logs every second):
```
python vts_face_tracker.py --monitor
```

**Monitor specific parameters**:
```
python vts_face_tracker.py --monitor --params FaceAngleX MouthOpen
```

**Set a custom parameter value**:
```
python vts_face_tracker.py --set MyCustomParam 0.5
```

## Recording

Record face tracking data for a set duration. Uses bulk reads (1 WebSocket call per frame) with auto-reconnect on connection drops.

**Record for 1 hour**:
```
python vts_face_tracker.py --record 1h
```

**Record for 30 minutes at 20 samples/sec, save as CSV**:
```
python vts_face_tracker.py --record 30m --sample-rate 20 --format csv
```

**Record directly as motion3.json**:
```
python vts_face_tracker.py --record 1h --format motion3 -o my_motion.motion3.json
```

**Record specific parameters only**:
```
python vts_face_tracker.py --record 10m --params FaceAngleX MouthOpen EyeOpenLeft
```

### Recording controls

| Key | Action |
|---|---|
| `P` | Pause/resume recording |
| `Ctrl+C` | Stop and save |

### Duration formats

| Format | Meaning |
|---|---|
| `30s` | 30 seconds |
| `5m` | 5 minutes |
| `1h` | 1 hour |
| `1h30m` | 1 hour 30 minutes |
| `1.5h` | 1.5 hours |
| `1h15m30s` | 1 hour 15 minutes 30 seconds |

### Recording options

| Option | Default | Description |
|---|---|---|
| `--record DURATION` | - | Duration to record |
| `--sample-rate N` | 10 | Samples per second |
| `--format json\|csv\|motion3` | json | Output format |
| `-o FILE` | auto | Output filename |
| `--params ...` | all | Specific parameters to record |

### Tracking loss handling

If the camera disconnects or face tracking drops to 0% confidence, those frames are automatically marked as lost. On export, lost frames are **cosine-interpolated** from surrounding good frames for smooth transitions. The progress bar shows a `Lost:` count in real-time.

## Playback & Injection

Play back a recorded file or inject it into VTS to drive your model.

**Inject into VTS (default)** — drives your Live2D model:
```
python vts_face_tracker.py --playback recording.json
python vts_face_tracker.py --playback recording.json --speed 2
python vts_face_tracker.py --playback recording.json --loop
```

**Display only (no injection)**:
```
python vts_face_tracker.py --playback recording.json --no-inject
```

**Playback from motion3.json** — auto-converts and caches:
```
python vts_face_tracker.py --playback my_motion.motion3.json
```

On first playback of a motion3.json, it converts to frames and saves a `.frames.json` cache for instant loading next time.

### Injection options

| Option | Default | Description |
|---|---|---|
| `--inject` | on | Inject data into VTS model |
| `--no-inject` | - | Display values only |
| `--speed N` | 1.0 | Playback speed multiplier |
| `--loop` | off | Loop continuously |

## Neural Network Motion Generator

Train a neural network on motion recordings and generate smooth VTuber motion autonomously. Supports style/emotion selection and long-duration generation without looping or running out of data.

### How it works

- **Train** an LSTM (CPU) or TCN (AMD GPU via DirectML) on your motion recordings
- **Generate** by following real recordings sequentially (natural human motion) with NN variation, smooth blinks, and crossfades between different files
- **Auto-padding**: if a style's files are too short for the requested duration, hour-long recordings and other files are added automatically, so the seed never runs out
- **Jitter-free**: raw tracking recordings are Gaussian-smoothed before use, and the NN output is EMA-filtered
- **No looping**: generation walks through seed recordings sequentially and crossfades into a random new region at file boundaries

### CPU Training (LSTM)

```
python neural_tracker.py train --data-dir "neuro network" --epochs 200 --fps 30
```

### GPU Training (AMD DirectML, Python 3.12)

```
venv312\Scripts\python.exe neural_tracker_gpu.py train --data-dir "neuro network" --epochs 200 --fps 30
```

GPU training uses a TCN (Temporal Convolutional Network) with balanced sampling so all styles get equal training time. Auto-scales batch size for high FPS training (70+ fps).

### Generating Motion

**Generate with a specific style**:
```
python neural_tracker.py generate --style dance --duration 30 --inject
```

**Generate with a random style** (never repeats the same style twice):
```
python neural_tracker.py generate --random --inject --duration 30
```

**Long-duration generation** (auto-pads with hour-long recordings):
```
python neural_tracker.py generate --random --inject --duration 900
```

**Generate and save to file**:
```
python neural_tracker.py generate --style idle --duration 60 -o my_motion.motion3.json
```

**Use GPU-trained model**:
```
python neural_tracker.py generate --model motion_model_gpu.pt --style dance --inject
```

### Available Styles

```
python neural_tracker.py list-styles
```

Styles: `idle`, `dance`, `angry`, `sad`, `wink`, `look_left`, `look_right`, `thinking`, `confused`, `happy`, `tracking`

### Neural Network Options

| Option | Default | Description |
|---|---|---|
| `--style STYLE` | idle | Motion style to generate |
| `--random` | off | Pick a random style (differs from last) |
| `--duration N` | 10 | Duration in seconds |
| `--fps N` | 30 | Output framerate |
| `--speed N` | 1.0 | Playback speed |
| `--inject` | off | Inject into VTS (streaming, no startup delay) |
| `-o FILE` | - | Save as motion3.json |
| `--model FILE` | auto | Model file to use |

### Generation features

- **Streaming injection** — generates and injects in 1-second batches, starts immediately even for very long durations
- **Smooth blinking** — natural 3-frame close/hold/open cycle every 4-8 seconds
- **Seed auto-padding** — short styles get hour-long recordings + other files added for long durations
- **Crossfades** — 30-frame smooth blends between different seed regions
- **Anti-loop** — sequential seed walking with persistent position across batches

## Idle Animation Export

Convert any recording into a VTS-ready idle animation (60 FPS, looped, smoothed, gap-interpolated).

```
python vts_face_tracker.py --idle recording.json
python vts_face_tracker.py --idle recording.json --trim 30s-35s
python vts_face_tracker.py --idle recording.json --trim 1m-1m30s --smooth-sigma 3
python vts_face_tracker.py --idle my_motion.motion3.json -o my_idle.motion3.json
```

### What `--idle` does

1. Loads any recording (JSON or motion3.json)
2. **Trims** to a time range (`--trim 30s-35s`)
3. **Filters** out parameters with no movement
4. **Interpolates** over tracking loss gaps
5. **Smooths** curves with Gaussian convolution
6. **Normalizes** values to fit Live2D output ranges
7. **Exports** at 60 FPS with `Loop: true` — ready for VTS

### Idle options

| Option | Default | Description |
|---|---|---|
| `--trim RANGE` | all | Time range (e.g. `30s-35s`, `1m-1m30s`) |
| `--smooth-sigma N` | 0.5 | Gaussian smoothing (0 = off) |
| `--keep-all` | off | Don't filter constant params |
| `-o FILE` | `*.idle.motion3.json` | Output path |

### Using in VTS

1. Copy the `.idle.motion3.json` file to your model's animation folder
2. In VTS Model Settings, set it as **"Default Idle Animation"**

## Convert

Convert a JSON recording to motion3.json:

```
python vts_face_tracker.py --convert recording.json
python vts_face_tracker.py --convert recording.json -o output.motion3.json
python vts_face_tracker.py --convert recording.json --motion3-loop
```

## GPU Batch Processing

AMD GPU-accelerated post-processing via OpenCL (`gpu_process.py`):

```
python gpu_process.py smooth recording.json -o smoothed.json --sigma 3
python gpu_process.py smooth recording.json -o filtered.json --method butterworth
python gpu_process.py interpolate recording.json -o upsampled.json --target-fps 60
python gpu_process.py simplify recording.json -o simplified.json --epsilon 0.02
python gpu_process.py transform recording.json -o scaled.json --scale 0.5 --offset 0.1
python gpu_process.py chain recording.json -o final.json --pipeline '[{"op":"smooth","sigma":2},{"op":"interpolate","target_fps":60}]'
```

| Command | What it does |
|---|---|
| `smooth` | Gaussian / Butterworth / Median filter |
| `interpolate` | Upsample to higher framerate |
| `simplify` | Reduce keyframes (Ramer-Douglas-Peucker) |
| `transform` | Offset, scale, clamp values |
| `chain` | Run multiple ops in sequence |

## Output formats

### JSON
```json
{
  "version": 1,
  "model": "My Model",
  "duration_target": 3600,
  "sample_rate": 10,
  "parameters": ["FaceAngleX", "MouthOpen", ...],
  "frame_count": 36000,
  "frames": [
    {"t": 1691098200.123, "frame": 0, "face": true, "FaceAngleX": 5.05, ...},
    ...
  ]
}
```

### motion3.json (Live2D)
```json
{
  "Version": 3,
  "Meta": {
    "Duration": 10.0,
    "Fps": 60,
    "Loop": true,
    "CurveCount": 3,
    "TotalSegmentCount": 297,
    "TotalPointCount": 300
  },
  "Curves": [
    {
      "Target": "Parameter",
      "Id": "ParamAngleX",
      "Segments": [0.0, 0.0, 0, 0.017, 0.1, 0, ...]
    }
  ]
}
```

## First Run

On first run, a popup will appear in VTube Studio asking to allow the plugin. Click "Allow". The authentication token is saved to `vts_token.txt` for future sessions.

## Requirements

- VTube Studio running with API enabled
- Python 3.10+
- websockets
- rich (optional, for colored display)
- numpy (for `--idle` export and neural network)
- torch (for neural network: CPU version with `pip install torch`, or GPU with `pip install torch-directml` in Python 3.12 venv)
- pyopencl (for GPU batch processing)

## Neural Network Setup (AMD GPU)

GPU training requires Python 3.12 (torch-directml doesn't support 3.14):

```
py -3.12 -m venv venv312
venv312\Scripts\pip install torch-directml pyopencl numpy websockets rich
```

The GPU model (`motion_model_gpu.pt`) can be used for generation with the regular Python interpreter:

```
python neural_tracker.py generate --model motion_model_gpu.pt --style dance --inject
```
