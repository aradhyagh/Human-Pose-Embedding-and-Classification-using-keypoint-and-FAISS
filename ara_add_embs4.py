from ara_loading_and_detecting3 import add_embeddings_from_new_root

model_path = r"F:\keypoints vector db\weights\yolov7-w6-pose (1).pt"
path_prefix = "pose_index"

# Add new dataset
add_embeddings_from_new_root(
    new_images_root=r"F:\keypoints vector db\raw images\add data to db",
    model_path=model_path,
    path_prefix=path_prefix
)