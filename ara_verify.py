from ara_integrate_2 import load_vector_db

path_prefix = 'pose_index'
index, labels = load_vector_db(path_prefix)
print("Number of vectors in index and labels:")
print(index.ntotal, len(labels))
