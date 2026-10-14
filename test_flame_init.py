import torch
import numpy as np
from types import SimpleNamespace

from submodules.flame_lib.FLAME import FLAME, FLAMETex

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Device:", device)

cfg = SimpleNamespace(
    flame_model_path="./models/FLAME2020/generic_model.pkl",
    flame_lmk_embedding_path="./models/landmark_embedding.npy",
    tex_space_path="./models/FLAME_albedo_from_BFM.npz",
    tex_type="BFM",
    shape_params=300,
    expression_params=100,
    tex_params=50,
)

print("Loading FLAME...")
flame = FLAME(cfg).to(device)

print("Loading FLAME texture...")
flametex = FLAMETex(cfg).to(device)

print("Creating parameters...")

shape = torch.zeros((1, 300), device=device)
expression = torch.zeros((1, 100), device=device)
jaw = torch.zeros((1, 3), device=device)

print("Running FLAME forward...")

with torch.no_grad():
    vertices, landmarks2d, landmarks3d = flame(
        shape_params=shape,
        expression_params=expression,
        jaw_pose_params=jaw,
    )

print("Vertices:", vertices.shape)
print("Landmarks 2D:", landmarks2d.shape)
print("Landmarks 3D:", landmarks3d.shape)

texcode = torch.zeros((1, 50), device=device)

print("Running texture forward...")

with torch.no_grad():
    texture = flametex(texcode)

print("Texture:", texture.shape)

print("\nFLAME INITIALIZATION TEST: SUCCESS")
