import os
import numpy as np
import torch
import mediapipe as mp

from submodules.flame_lib.FLAME import FLAME

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

IMAGE_PATH = "../reconstruction/neutral.jpg"
FLAME_MODEL = "./models/FLAME2020/generic_model.pkl"
LANDMARK_EMBEDDING = "./models/landmark_embedding.npy"
OUTPUT_OBJ = "./mac_flame_head.obj"


def detect_landmarks(image_path):
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="./models/face_landmarker.task"
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    with FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError("No face detected")

    landmarks = result.face_landmarks[0]

    h, w = image.shape[:2]

    points = np.array(
        [[p.x * w, p.y * h] for p in landmarks],
        dtype=np.float32,
    )

    print("Image:", image.shape)
    print("MediaPipe landmarks:", points.shape)

    return points


def main():
    print("Device:", DEVICE)

    # ---------------------------------------------------------
    # 1. MediaPipe
    # ---------------------------------------------------------
    points = detect_landmarks(IMAGE_PATH)

    # ---------------------------------------------------------
    # 2. FLAME
    # ---------------------------------------------------------
    print("Loading FLAME...")

    flame = FLAME(
        flame_model_path=FLAME_MODEL,
        landmark_embedding_path=LANDMARK_EMBEDDING,
        batch_size=1,
    ).to(DEVICE)

    flame.eval()

    print("FLAME loaded")

    # ---------------------------------------------------------
    # 3. Neutral FLAME mesh
    # ---------------------------------------------------------
    shape = torch.zeros(
        (1, 300),
        dtype=torch.float32,
        device=DEVICE,
    )

    expression = torch.zeros(
        (1, 100),
        dtype=torch.float32,
        device=DEVICE,
    )

    pose = torch.zeros(
        (1, 6),
        dtype=torch.float32,
        device=DEVICE,
    )

    neck_pose = torch.zeros(
        (1, 3),
        dtype=torch.float32,
        device=DEVICE,
    )

    jaw_pose = torch.zeros(
        (1, 3),
        dtype=torch.float32,
        device=DEVICE,
    )

    eyes_pose = torch.zeros(
        (1, 6),
        dtype=torch.float32,
        device=DEVICE,
    )

    transl = torch.zeros(
        (1, 3),
        dtype=torch.float32,
        device=DEVICE,
    )

    print("Generating FLAME mesh...")

    with torch.no_grad():
        vertices, landmarks = flame(
            shape_params=shape,
            expression_params=expression,
            pose_params=pose,
            neck_pose=neck_pose,
            jaw_pose=jaw_pose,
            eyes_pose=eyes_pose,
            transl=transl,
        )

    vertices = vertices[0].detach().cpu().numpy()

    print("FLAME vertices:", vertices.shape)

    # ---------------------------------------------------------
    # 4. Export OBJ
    # ---------------------------------------------------------
    faces = flame.faces_tensor.detach().cpu().numpy()

    with open(OUTPUT_OBJ, "w") as f:
        for v in vertices:
            f.write(
                f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n"
            )

        for face in faces:
            f.write(
                f"f {face[0] + 1} "
                f"{face[1] + 1} "
                f"{face[2] + 1}\n"
            )

    print()
    print("SUCCESS")
    print("OBJ:", OUTPUT_OBJ)


if __name__ == "__main__":
    main()
