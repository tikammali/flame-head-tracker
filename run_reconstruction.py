import torch
from tracker_base import Tracker

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", device)

tracker_cfg = {
    "mediapipe_face_landmarker_v2_path": "./models/face_landmarker.task",
    "flame_model_path": "./models/FLAME2020/generic_model.pkl",
    "flame_lmk_embedding_path": "./models/landmark_embedding.npy",
    "ear_landmarker_path": "./models/ear_landmarker.pth",
    "tex_space_path": "./models/FLAME_albedo_from_BFM.npz",
    "face_parsing_model_path": "./models/79999_iter.pth",
    "template_mesh_file_path": "./models/head_template.obj",

    "result_img_size": 512,
    "use_matting": False,
    "optimize_fov": True,
    "device": device,
}

print("Creating Tracker...")
tracker = Tracker(tracker_cfg)

print("Tracker initialized successfully.")

print("Running reconstruction...")

result = tracker.load_image_and_run(
    "../reconstruction/neutral.jpg",
    realign=True,
    photometric_fitting=False,
)

print("Reconstruction completed.")

print("Result keys:")
for key in result:
    value = result[key]
    if hasattr(value, "shape"):
        print(f"  {key}: {value.shape}")
    else:
        print(f"  {key}: {type(value)}")
