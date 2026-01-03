from ara_integrate_2 import load_vector_db,save_vector_db, add_embeddings_to_db, search_top_n
from ara_kpts import run_yolov7_pose, extract_all_kpts, make_pose_embedding

# Load index later

import os
import numpy as np
from tqdm import tqdm
import time

def add_embeddings_from_new_root(
    new_images_root,
    model_path,
    path_prefix,
    embedding_dim=25
):
    """
    Adds new embeddings from a new root folder (same class subfolders)
    to an existing FAISS vector DB and saves it.

    Args:
        new_images_root (str): New dataset root folder
        model_path (str): YOLOv7 pose model path
        path_prefix (str): Saved FAISS DB prefix
        embedding_dim (int): Embedding dimension (25)

    Returns:
        updated_index, updated_labels
    """

    # Load existing DB
    index, labels = load_vector_db(path_prefix)

    new_embeddings = []
    new_labels = []

    # Count total images
    total_images = sum(
        len(files)
        for _, _, files in os.walk(new_images_root)
    )

    print(f"Adding {total_images} new images to vector DB...")
    start_time = time.time()

    with tqdm(total=total_images, desc="Adding embeddings") as pbar:
        for label in os.listdir(new_images_root):
            class_dir = os.path.join(new_images_root, label)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                img_path = os.path.join(class_dir, fname)
                try:
                    output = run_yolov7_pose(img_path, model_path)
                    if len(output) == 0:
                        pbar.update(1)
                        continue

                    kpts = extract_all_kpts(output[0])
                    emb = make_pose_embedding(kpts)

                    new_embeddings.append(emb)
                    new_labels.append(label)

                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

                pbar.update(1)

    if len(new_embeddings) == 0:
        print("No new embeddings added.")
        return index, labels

    new_embeddings = np.array(new_embeddings, dtype="float32")
    index.add(new_embeddings)
    labels.extend(new_labels)

    save_vector_db(index, labels, path_prefix)

    elapsed = time.time() - start_time
    print(f"Finished adding embeddings in {int(elapsed//60)}m {int(elapsed%60)}s")

    return index, labels

def query_image_pose(
    image_path,
    model_path,
    path_prefix,
    top_k=5
):
    """
    Queries FAISS DB with an image and returns prediction results.

    Args:
        image_path (str): Query image path
        model_path (str): YOLOv7 pose model
        path_prefix (str): FAISS DB prefix
        top_k (int): Number of nearest neighbors

    Returns:
        predicted_label, nearest_labels, distances
    """

    # Load DB
    index, labels = load_vector_db(path_prefix)

    # Create embedding
    output = run_yolov7_pose(image_path, model_path)
    if len(output) == 0:
        raise ValueError("No person detected in query image")

    kpts = extract_all_kpts(output[0])
    query_emb = make_pose_embedding(kpts)

    # Search
    predicted_label, nearest_labels, distances = search_top_n(
        index, labels, query_emb, n=top_k
    )

    return predicted_label, nearest_labels, distances
