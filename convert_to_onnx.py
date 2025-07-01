import os
import torch
import onnx
from google.protobuf.message import EncodeError
from onnx import external_data_helper
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from optimum.onnx.graph_transformations import check_and_save_model

# === Constants ===
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = "output/onnx"
TMP_ONNX = os.path.join(OUTPUT_DIR, "vision_encoder.onnx.tmp")
FINAL_ONNX = os.path.join(OUTPUT_DIR, "vision_encoder.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load model + processor ===
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# === Make dummy vision inputs ===
vcfg = model.config.vision_config
batch_size = 1
grid_t, grid_h, grid_w = 1, 16, 16

# spatial_patch_size is an int, so we square it
patch_size = vcfg.spatial_patch_size
feat_dim = vcfg.in_chans * vcfg.temporal_patch_size * (patch_size ** 2)

pixel_values = torch.randn(
    batch_size * grid_t * grid_h * grid_w,
    feat_dim,
    dtype=torch.float32,
)
grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64)

# === 1) Export the vision encoder to a .tmp ONNX file ===
torch.onnx.export(
    model.visual,
    args=(pixel_values, grid_thw),
    f=TMP_ONNX,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["pixel_values", "grid_thw"],
    output_names=["image_features"],
    dynamic_axes={
        "pixel_values":   {0: "batch_size * grid_t * grid_h * grid_w"},
        "grid_thw":       {0: "batch_size"},
        "image_features": {0: "batch_size * grid_t * grid_h * grid_w"},
    },
)

# === 2) Load it back in and save, with an external-data fallback if >2 GB ===
model_proto = onnx.load(TMP_ONNX)

try:
    # This will throw EncodeError if the proto is too large
    check_and_save_model(model_proto, FINAL_ONNX)
except EncodeError:
    print("Proto too large — spilling weights to external data…")
    external_data_helper.convert_model_to_external_data(
        model_proto,
        all_tensors_to_one_file=True,
        location="vision_encoder.onnx.data",
        size_threshold=1024  # 1 KB → effectively everything
    )
    onnx.save_model(model_proto, FINAL_ONNX)

# === Cleanup ===
os.remove(TMP_ONNX)
print(f"✅ Exported vision encoder to:\n  {FINAL_ONNX}\n"
      f"  (plus {FINAL_ONNX}.data if created)")
