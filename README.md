# VTube Studio Face Tracker

A Python client for recording, playing back, and processing face tracking data from VTube Studio via its WebSocket API. Supports AMD GPU acceleration for batch processing via OpenCL.

## Setup

1. Enable **"Allow Plugin API access"** in VTube Studio settings
2. Install dependencies:
```
pip install -r requirements.txt
```
For GPU batch processing:
```
pip install pyopencl numpy
```

## Files

| File | Description |
|---|---|
| `vts_face_tracker.py` | Main tracker (same as amd_gpu version) |
| `vts_face_tracker_amd_gpu.py` | AMD GPU copy (identical) |
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

If the camera disconnects or face tracking drops to 0% confidence, those frames are automatically marked as lost. On export, lost frames are **linearly interpolated** from surrounding good frames for smooth transitions. The progress bar shows a `Lost:` count in real-time.

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
6. **Exports** at 60 FPS with `Loop: true` — ready for VTS

### Idle options

| Option | Default | Description |
|---|---|---|
| `--trim RANGE` | all | Time range (e.g. `30s-35s`, `1m-1m30s`) |
| `--smooth-sigma N` | 2.0 | Gaussian smoothing (0 = off) |
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
- numpy (for `--idle` export)
- pyopencl (for GPU batch processing)
