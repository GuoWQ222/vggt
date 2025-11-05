import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import json
from iopath.common.file_io import g_pathmgr

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT()
ckpt_path = "/mnt/shared-storage-user/guowenqi/codebase/vggt/vggt.pt"
print(f"Resuming training from {ckpt_path} ")

with g_pathmgr.open(ckpt_path, "rb") as f:
    checkpoint = torch.load(f, map_location="cpu")

# Load model state
model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
missing, unexpected = model.load_state_dict(
    model_state_dict, strict=False
)
print(f"Missing keys when loading pretrained model: {missing[:10]}")
print(f"Unexpected keys when loading pretrained model: {unexpected[:10]}")
model.to(device)


# Load and preprocess example images (replace with your own image paths)
folder = "/mnt/shared-storage-user/idc2-shared/dataset/preprocess/hypersim_processed/ai_001_001/cam_00"
json_file = "/cpfs/user/guowenqi/dataset/co3dv2/apple/selected_seqs_test.json"
# with open(json_file, 'r') as f:
#     data = json.load(f)
#     data = data["110_13051_23361"]
# image_path = []
# for id in data: 
#     fname = f"frame{id:06d}.jpg"
#     image_path.append(os.path.join(folder, fname))
image_path = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(('.jpg','.png'))]
image_path.sort()
image_path = image_path[::5]
images = load_and_preprocess_images(image_path).to(device)
print(images.shape)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        print(predictions)