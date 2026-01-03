import torch
import cv2
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


def run_yolov7_pose(image_path: str, model_path: str):
    """
    Runs YOLOv7 pose model on a single image.

    Args:
        image_path (str): Path to input image
        model_path (str): Path to YOLOv7 pose model (.pt)

    Returns:
        output (np.ndarray): Keypoint output after NMS
    """

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    weights = torch.load(model_path, map_location=device, weights_only=False)
    model = weights['model']
    model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)

    # Inference
    with torch.no_grad():
        output, _ = model(image)

        output = non_max_suppression_kpt(
            output,
            conf_thres=0.25,
            iou_thres=0.65,
            nc=model.yaml['nc'],
            nkpt=model.yaml['nkpt'],
            kpt_label=True
        )

        output = output_to_keypoint(output)

    return output

def extract_all_kpts(det_row):
    """
    Extracts all 17 COCO keypoints from a YOLOv7-Pose detection row.

    Args:
        det_row (array-like): One row from output_to_keypoint()

    Returns:
        tuple: 17 keypoints in order, each as (x, y, conf)
    """

    kpt_names = (
        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    )

    # Keypoints start at index 7
    start_idx = 7

    kpts = []
    for i in range(17):
        x = float(det_row[start_idx + i * 3])
        y = float(det_row[start_idx + i * 3 + 1])
        c = float(det_row[start_idx + i * 3 + 2])
        kpts.append((x, y, c))

    return tuple(kpts)

import math
import numpy as np


def _distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _angle(a, b, c):
    """
    Angle at point b for points a-b-c
    Returns angle in radians
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.acos(cos_angle)


def make_pose_embedding(kpts):
    """
    Args:
        kpts (tuple): 17 keypoints as (x, y, conf)

    Returns:
        list: 25D pose embedding
    """

    # Named access for clarity
    (
        nose,
        l_eye, r_eye,
        l_ear, r_ear,
        l_shoulder, r_shoulder,
        l_elbow, r_elbow,
        l_wrist, r_wrist,
        l_hip, r_hip,
        l_knee, r_knee,
        l_ankle, r_ankle
    ) = kpts

    embedding = []

    # -------------------------------
    # 0–15: Normalized distances from nose
    # -------------------------------
    distances = []
    for kp in kpts[1:]:  # exclude nose itself
        distances.append(_distance(nose, kp))

    max_dist = max(distances) + 1e-6
    distances = [d / max_dist for d in distances]
    embedding.extend(distances)

    # -------------------------------
    # 16: Angle l_shoulder–nose–r_shoulder
    # -------------------------------
    embedding.append(
        _angle(l_shoulder, nose, r_shoulder) / math.pi
    )

    # -------------------------------
    # 17–24: Joint angles (normalized)
    # -------------------------------
    angle_defs = [
        (l_shoulder, l_elbow, l_wrist),     # 17
        (r_shoulder, r_elbow, r_wrist),     # 18
        (l_hip, l_shoulder, l_elbow),       # 19
        (r_hip, r_shoulder, r_elbow),       # 20
        (l_shoulder, l_hip, l_knee),        # 21
        (r_shoulder, r_hip, r_knee),        # 22
        (l_hip, l_knee, l_ankle),            # 23
        (r_hip, r_knee, r_ankle)             # 24
    ]

    for a, b, c in angle_defs:
        embedding.append(_angle(a, b, c) / math.pi)

    return embedding

########################################
#
# output = run_yolov7_pose(
#     image_path=r"F:\keypoints vector db\images_data\3 person.jpg",
#     model_path=r"F:\keypoints vector db\weights\yolov7-w6-pose (1).pt"
# )
# # For first detected person
# person_kpts = extract_all_kpts(output[0])
#
# embedding = make_pose_embedding(person_kpts)
# print(embedding)
# print(len(embedding))  # ✅ 25
# print(person_kpts)
#
