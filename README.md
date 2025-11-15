# Real-Time Lane Detection in Unity

> End-to-end Unity project that renders a camera feed, runs a lane-segmentation model, and overlays a translucent mask in real time.

![Lane detection demo](recordings/lane_det.gif)

---

## Features

- **Unity Inference Engine** worker (GPUCompute or CPU) running an ONNX model
- **Camera → Tensor → Mask** pipeline at fixed input size (e.g., 640×384)
- **UI overlay** via `RawImage` + simple transparent material
- **Driveable car** with front camera for nn input

---

## How It Works

1. **Render camera** to a low-res `RenderTexture` matching the model input (e.g., 640×384).
2. **Read pixels** to `Texture2D` and **pack NHWC floats** with normalization (e.g., `[-1,1]`).
3. **Schedule inference** with a `Worker` and **peek output** by layer/tensor name (e.g., `"conv2d_29"`).
4. **Threshold** probabilities to build a **pink RGBA mask** (`Texture2D`) with alpha.
5. **Overlay** the mask in the UI via **`RawImage`**.

---

## Scripts Overview

- **`laneDetection.cs`** — Camera→Tensor→Mask loop, ONNX runtime, UI overlay
- **`CarControl.cs`** — Rigidbody car controller using wheel colliders, speed-scaled torque/steer, braking logic
- **`WheelControl.cs`** — Syncs visible wheel mesh pose with the collider each frame
- **`CarInputActions.cs` / `.inputactions`** — Input System (WASD as a 2D Vector for movement)

---

## Notebooks

- `Notebooks/canny_edge.ipynb` — quick OpenCV baseline
- `Notebooks/cnn_lane_det.ipynb` — training/inference playground for a lightweight lane CNN
