# Human-Pose-Embedding-and-Classification-using-keypoint-and-FAISS

This project recognizes and compares human postures (such as standing, sitting, running, etc.) from images by analyzing body keypoints and storing them as embeddings in a vector database.

---

## What this project does (in simple terms)

- Looks at an image of a person
- Detects important body joints (head, arms, legs, etc.)
- Converts the body posture into a numerical representation (embedding)
- Stores these embeddings in a database
- Compares new images with stored ones to predict the posture

---

## Dataset Structure

images_data/

├── standing/

│ ├── img1.jpg

│ ├── img2.jpg

├── sitting/

│ ├── img3.jpg

│ ├── img4.jpg

├── running/

│ ├── img5.jpg


Each subfolder name is treated as the **label/class**.

---

## Core Pipeline

1. **Pose Detection**
   - Uses YOLOv7-Pose to detect 17 human body keypoints per person.

2. **Embedding Creation**
   - Converts keypoints into a 25-dimensional embedding:
     - Distances of joints from the nose
     - Important joint angles (arms, legs, shoulders)
     - All values normalized for consistency

3. **Vector Database**
   - Stores embeddings in a FAISS vector index
   - Labels are stored alongside embeddings

4. **Querying**
   - Given a new image, the system:
     - Extracts pose
     - Creates an embedding
     - Finds the most similar poses in the database
     - Predicts the posture label

---

## Tech Stack

- Python
- PyTorch
- YOLOv7-Pose
- OpenCV
- NumPy
- FAISS
- tqdm (progress bars)

---

## Features

- Automatic dataset traversal
- Pose-based embedding generation
- Fast similarity search using FAISS
- Incremental database updates
- Progress bar with time tracking
- Works across multiple scripts

---

## Saved Files

After building the database, two files are created:

- `pose_index.faiss` → FAISS vector database
- `pose_index_labels.pkl` → Corresponding labels

---

## Example Use Cases

- Human posture classification
- Activity recognition (standing, sitting, running)
- Pose similarity search
- Fitness or sports analysis
- Surveillance posture analysis

---

## Notes

- The system currently uses the **first detected person** in each image.
- Designed to be modular and reusable.
- Can be extended to videos, confidence filtering, or large-scale datasets.

---

## Author

Built as a learning and experimentation project to understand pose-based embeddings and similarity search.

---

⭐ If you find this useful, feel free to star the repository!
