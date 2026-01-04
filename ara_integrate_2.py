import os
import time
import numpy as np
import faiss
import pickle
from collections import Counter
from tqdm import tqdm  # progress bar

# Import your existing functions
from ara_kpts1 import run_yolov7_pose, extract_all_kpts, make_pose_embedding

# ----------------------------
# Helper to format elapsed time
# ----------------------------
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h:{m}m:{s}s"

# ----------------------------
# 1. Build FAISS index from folder with progress bar
# ----------------------------
def build_faiss_index_from_folder(images_root, model_path, embedding_dim=25):
    """
    Traverses folder and builds FAISS index of embeddings + labels.
    Assumes subfolders = class labels.
    Shows progress bar with estimated time.
    """
    embeddings_list = []
    labels_list = []

    # Count total images for progress bar
    total_images = 0
    for label in os.listdir(images_root):
        class_folder = os.path.join(images_root, label)
        if not os.path.isdir(class_folder):
            continue
        total_images += len(os.listdir(class_folder))

    print(f"Total images to process: {total_images}")
    start_time = time.time()

    with tqdm(total=total_images, desc="Processing images") as pbar:
        for label in os.listdir(images_root):
            class_folder = os.path.join(images_root, label)
            if not os.path.isdir(class_folder):
                continue
            for fname in os.listdir(class_folder):
                fpath = os.path.join(class_folder, fname)
                try:
                    output = run_yolov7_pose(fpath, model_path)
                    if len(output) == 0:
                        pbar.update(1)
                        continue
                    kpts = extract_all_kpts(output[0])
                    emb = make_pose_embedding(kpts)
                    embeddings_list.append(emb)
                    labels_list.append(label)
                except Exception as e:
                    print(f"Skipping {fpath}, error: {e}")
                pbar.update(1)

    elapsed = time.time() - start_time
    print(f"Finished processing images in {format_time(elapsed)}")

    embeddings_array = np.array(embeddings_list, dtype='float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_array)

    return index, embeddings_array, labels_list

# ----------------------------
# 2. Save FAISS index + labels
# ----------------------------
def save_vector_db(index, labels, path_prefix):
    """
    Saves FAISS index and labels to disk
    """
    faiss.write_index(index, path_prefix + ".faiss")
    with open(path_prefix + "_labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    print(f"Saved FAISS index and labels to {path_prefix}.*")

# ----------------------------
# 3. Load FAISS index + labels
# ----------------------------
def load_vector_db(path_prefix):
    """
    Loads FAISS index and labels from disk
    """
    index = faiss.read_index(path_prefix + ".faiss")
    with open(path_prefix + "_labels.pkl", "rb") as f:
        labels = pickle.load(f)
    return index, labels

# ----------------------------
# 4. Add new embeddings + labels and rewrite DB
# ----------------------------
def add_embeddings_to_db(index, labels, new_embeddings, new_labels, path_prefix):
    """
    Adds new embeddings + labels to existing FAISS index and saves
    """
    new_embeddings = np.array(new_embeddings, dtype='float32')
    index.add(new_embeddings)
    labels.extend(new_labels)
    save_vector_db(index, labels, path_prefix)
    return index, labels

# ----------------------------
# 5. Search top-N matches
# ----------------------------
def search_top_n(index, labels, query_embedding, n=5):
    """
    Search for top-N nearest neighbors and return predicted label
    """
    query_embedding = np.array([query_embedding], dtype='float32')
    D, I = index.search(query_embedding, n)
    nearest_labels = [labels[i] for i in I[0]]
    predicted_label = Counter(nearest_labels).most_common(1)[0][0]
    return predicted_label, nearest_labels, D[0]

######################

images_root = r"F:\keypoints vector db\images_data"
model_path = r"F:\keypoints vector db\weights\yolov7-w6-pose (1).pt"

# # Build FAISS index with progress bar
# index, embeddings_array, labels_list = build_faiss_index_from_folder(images_root, model_path)
#
# # Save index + labels
# save_vector_db(index, labels_list, path_prefix)
#

# print(labels_list)
