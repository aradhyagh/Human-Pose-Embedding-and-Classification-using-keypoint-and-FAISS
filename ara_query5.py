from ara_loading_and_detecting3 import query_image_pose
model_path = r"F:\keypoints vector db\weights\yolov7-w6-pose (1).pt"
path_prefix = "pose_index"

# Query image
pred_label, neighbors, dists = query_image_pose(
    image_path=r"F:\keypoints vector db\raw images\7.jpg",
    model_path=model_path,
    path_prefix=path_prefix,
    top_k=5
)

print("Predicted:", pred_label)
print("Nearest:", neighbors)
print("Distances:", dists)