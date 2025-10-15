import nibabel as nib
import numpy as np
from sklearn.metrics import adjusted_rand_score
import glob
import os
from scipy.ndimage import zoom

# Paths for parcellations and masks
parcellation_dir = r"E:/INM-7/SuperCBP/code/calc_ARI"
mask_dir = r"E:/INM-7/SuperCBP/final masks"

# List of parcellation files
parcellation_files = sorted(glob.glob(os.path.join(parcellation_dir, "Amygdala_cluster_*_right_GA.nii.gz" )))

# List of mask files
mask_files = [
    "Tian-lAMY-rh.nii.gz",
    "Tian-mAMY-rh.nii.gz"
]
mask_files = [os.path.join(mask_dir, f) for f in mask_files]

# Create label map for each parcellation
def create_label_map(parcellation_files):
    label_map = None
    for idx, f in enumerate(parcellation_files):
        img = nib.load(f)
        data = img.get_fdata()
        if label_map is None:
            label_map = np.zeros(data.shape, dtype=int)
        # Assign a unique label to each cluster
        label_map[data > 0] = idx + 1  # Labels start from 1
    return label_map

# Create label map for masks
def create_mask_label_map(mask_files):
    label_map = None
    for idx, f in enumerate(mask_files):
        img = nib.load(f)
        data = img.get_fdata()
        if label_map is None:
            label_map = np.zeros(data.shape, dtype=int)
        # Assign a unique label to each mask
        label_map[data > 0] = idx + 1
    return label_map

def resample_to_shape(data, target_shape):
    factors = [t / s for t, s in zip(target_shape, data.shape)]
    return zoom(data, factors, order=0)  # nearest-neighbor for labels


# Generate label maps
parcellation_label_map = create_label_map(parcellation_files)
mask_label_map = create_mask_label_map(mask_files)

# Resample mask_label_map if shapes do not match
if parcellation_label_map.shape != mask_label_map.shape:
    print(f"Resampling mask from shape {mask_label_map.shape} to {parcellation_label_map.shape}")
    mask_label_map = resample_to_shape(mask_label_map, parcellation_label_map.shape)


# Select voxels that have a label in at least one of the maps
mask = (parcellation_label_map > 0) | (mask_label_map > 0)
labels1 = parcellation_label_map[mask].flatten()
labels2 = mask_label_map[mask].flatten()

# Calculate Adjusted Rand Index (ARI)
ari = adjusted_rand_score(labels1, labels2)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")