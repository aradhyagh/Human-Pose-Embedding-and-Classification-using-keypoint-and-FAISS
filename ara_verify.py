from ara_integrate_2 import load_vector_db

path_prefix = 'pose_index'
index, labels = load_vector_db(path_prefix)
print("Number of vectors in index and labels:")
print(index.ntotal, len(labels))

def value_counts_list(lst):
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    return counts

print('label counts: ', value_counts_list(labels))
