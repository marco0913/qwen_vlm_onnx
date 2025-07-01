
# 🧠 Real-Time Image Captioning with Qwen2-VL and ONNX

This project demonstrates how to:
1. Export the **vision encoder** from the [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) model to ONNX.
2. Use **ONNX Runtime** for fast vision feature extraction.
3. Run **real-time webcam inference** to describe what the camera sees.

---

## 📦 Requirements

Install the necessary packages (tested versions shown below):

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
.
├── convert_to_onnx.py          # Exports the vision encoder to ONNX
├── qwen_vlm_camera_live.py     # Real-time webcam-based captioning
├── output/
│   └── onnx/
│       ├── vision_encoder.onnx
│       └── vision_encoder.onnx.data  (if model >2GB)
└── requirements.txt
└── README.md
```

---

## 🚀 Step 1: Export Vision Encoder to ONNX

Run this once:

```bash
python export_onnx.py
```

This will:
- Export the `model.visual` component (vision encoder) to ONNX.
- Save it as `output/onnx/vision_encoder.onnx`.
- Automatically store weights in external data if the model is large.

---

## 🎥 Step 2: Run Real-Time Webcam Captioning

Start the live captioning:

```bash
python run_inference.py
```

- Captures frames from your webcam.
- Extracts image features using the ONNX model.
- Feeds the image into the **full PyTorch model** to generate captions.
