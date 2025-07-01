import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import cv2
import threading
import time

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ONNX_VISION_PATH = "output/onnx/vision_encoder.onnx"
CAMERA_ID = 0  # Change this to your camera ID
INFERENCE_INTERVAL = 2.0  # Seconds between inferences

# ─── GLOBALS ────────────────────────────────────────────────────────────────────
current_frame = None
frame_lock = threading.Lock()
running = True

# ─── HELPERS ────────────────────────────────────────────────────────────────────


def print_onnx_model_info(path: str):
    model = onnx.load(path)
    print("\n🧠 ONNX Model Inputs:")
    for input in model.graph.input:
        name = input.name
        dims = [d.dim_value if d.dim_value >
                0 else "?" for d in input.type.tensor_type.shape.dim]
        dtype = input.type.tensor_type.elem_type
        print(f" - {name}: shape={dims}, dtype={dtype}")

    print("\n🧪 ONNX Model Outputs:")
    for output in model.graph.output:
        name = output.name
        dims = [d.dim_value if d.dim_value >
                0 else "?" for d in output.type.tensor_type.shape.dim]
        dtype = output.type.tensor_type.elem_type
        print(f" - {name}: shape={dims}, dtype={dtype}")


def run_onnx_vision(frame: np.ndarray, processor, vision_sess):
    # Convert OpenCV BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image (note: processor expects PIL image but we'll simulate it)
    inputs = processor(text=[""], images=frame_rgb, return_tensors="np")
    onnx_inputs = {
        "pixel_values": inputs["pixel_values"],
        "grid_thw":     inputs["image_grid_thw"],
    }

    feats = vision_sess.run(["image_features"], onnx_inputs)[0]
    return feats


def generate_caption(frame: np.ndarray, model, processor):
    # Convert OpenCV BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[frame_rgb],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]


def camera_capture():
    global current_frame, running

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue

            with frame_lock:
                current_frame = frame.copy()

            # Display the frame (optional)
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def inference_loop(processor, vision_sess, model):
    global running

    last_inference_time = 0

    while running:
        current_time = time.time()
        if current_time - last_inference_time < INFERENCE_INTERVAL:
            time.sleep(0.1)
            continue

        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue

            frame = current_frame.copy()

        try:
            # Run ONNX vision encoder
            feats = run_onnx_vision(frame, processor, vision_sess)
            print(f"✔ ONNX image_features shape: {feats.shape}")

            # Generate caption
            caption = generate_caption(frame, model, processor)
            print(f"\n🖼 Caption: {caption}")

            last_inference_time = current_time

        except Exception as e:
            print(f"Error during inference: {e}")
            time.sleep(1)

# ─── MAIN ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    try:
        # Load models (this takes some time)
        print("Loading models...")
        print_onnx_model_info(ONNX_VISION_PATH)
        vision_sess = ort.InferenceSession(ONNX_VISION_PATH)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID).eval()
        print("Models loaded successfully!")

        # Start camera thread
        camera_thread = threading.Thread(target=camera_capture)
        camera_thread.start()

        # Run inference in main thread
        inference_loop(processor, vision_sess, model)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        running = False
        camera_thread.join()
        print("Program exited cleanly.")
