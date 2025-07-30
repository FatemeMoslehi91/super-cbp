import numpy as np
import nibabel as nib

# Path to the reference NIfTI image (used for shape and affine)
ref_nii_path = r'Tian_left_hippocampus_mask.nii.gz'  

# Path to the .npy file containing cluster assignments for each voxel
npy_path = r'E:/INM-7/SuperCBP/code/mask 2/clustering_results_for_mask/left_hemi_cluster_assignments_log.npy' 

# Load the reference NIfTI image
ref_img = nib.load(ref_nii_path)
ref_data = ref_img.get_fdata()  
shape = ref_data.shape          
affine = ref_img.affine         

# Load the cluster assignments for each voxel 
clusters = np.load(npy_path)


mask = ref_data > 0
voxel_indices = np.where(mask)  # Indices of voxels inside the mask

num_mask_voxels = np.sum(mask)      # Total number of voxels inside the mask
num_clusters = clusters.shape[0]   

# Loop over each cluster 
for cluster_num in range(4):
    mask_img = np.zeros(shape, dtype=np.uint8)  # Create an empty mask image
    mask_img[voxel_indices] = (clusters == cluster_num).astype(np.uint8)
    output_nii_path = f'test_cluster{cluster_num}_mask.nii.gz'
    mask_nii = nib.Nifti1Image(mask_img, affine) # Create a new NIfTI image for the mask
    nib.save(mask_nii, output_nii_path)
    print(f"Mask for cluster {cluster_num} saved to {output_nii_path}") 