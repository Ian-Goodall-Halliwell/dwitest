from nifreeze.data import dmri
from nifreeze.estimator import Estimator
import nibabel as nib
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import ants
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel
import os
import subprocess
import sys

def bias_field_correction(image, output):
    """
    Apply N4 bias field correction to the input image using the provided mask.

    Parameters:
    - image: str, path to the input image file.
    - output: str, path to save the corrected image.
    """
    img = ants.image_read(image)
    corrected_img = ants.n4_bias_field_correction(img)
    ants.image_write(corrected_img, output)

def process_dwi(input_dwi, bval_file, bvec_file, output_dir):
    """
    Process DWI data including denoising, motion correction, brain extraction, bias field correction, and tensor fitting.

    Parameters:
    - input_dwi: str, path to the input DWI NIfTI file.
    - bval_file: str, path to the bval file.
    - bvec_file: str, path to the bvec file.
    - output_dir: str, path to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load DWI NIfTI file
    dwi_img, dwi_affine = load_nifti(input_dwi)

    # Load b-values and b-vectors
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)

    # Create a gradient table
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)

    print("DWI shape:", dwi_img.shape)
    print("B-values:", np.unique(bvals))

    # Estimate noise standard deviation
    sigma = estimate_sigma(dwi_img, N=64)

    # Apply Non-Local Means (NL-Means) denoising
    denoised_dwi = nlmeans(dwi_img, sigma=sigma)

    # Save the denoised image
    denoised_dwi_path = os.path.join(output_dir, "dwi_denoised.nii.gz")
    nib.save(nib.Nifti1Image(denoised_dwi, dwi_affine), denoised_dwi_path)

    # Load the denoised DWI data
    dwi_data = dmri.load(
        denoised_dwi_path,
        bval_file=bval_file,
        bvec_file=bvec_file,
    )

    # Motion and eddy current correction
    estimator = Estimator(model="meandwi")
    estimated_affine = estimator.run(dwi_data)
    motion_corrected_path = os.path.join(output_dir, "dwi_motion_corrected.nii.gz")
    dwi_data.to_nifti(motion_corrected_path)

    # Update gradient table with corrected bvals and bvecs
    gtab = gradient_table(bvals=dwi_data.bvals, bvecs=dwi_data.bvecs)

    # Load the motion-corrected DWI data
    corrected_dwi = nib.load(motion_corrected_path)
    
    b0_masked, brain_mask = median_otsu(corrected_dwi.get_fdata(), vol_idx=[3],)
    # output_hdbet = os.path.join(output_dir, "HD_BET_" + os.path.basename(motion_corrected_path))
    # hd_bet_cmd = [
    #     "hd-bet",
    #     "-i", motion_corrected_path,  # input motion-corrected DWI
    #     "-o", output_hdbet,             # output directory
    #     "-device", "cpu"              # or "cuda" if using GPU
    # ]
    # subprocess.run(hd_bet_cmd, check=True)

    # # Assuming HD-BET outputs a brain-extracted image and mask with a standardized naming convention:
    # brain_masked_path = os.path.join(output_dir, "HD_BET_" + os.path.basename(motion_corrected_path))
    # brain_mask_path = os.path.join(
    #     output_dir,
    #     "HD_BET_" + os.path.splitext(os.path.basename(motion_corrected_path))[0] + "_mask.nii.gz"
    # )

    # # Load the outputs
    # b0_masked = nib.load(brain_masked_path).get_fdata()
    # brain_mask = nib.load(brain_mask_path).get_fdata().astype(np.int32)
    
    # Save brain-masked DWI and brain mask
    brain_masked_path = os.path.join(output_dir, "dwi_brain_masked.nii.gz")
    brain_mask_path = os.path.join(output_dir, "brain_mask.nii.gz")
    nib.save(nib.Nifti1Image(b0_masked, dwi_affine), brain_masked_path)
    nib.save(nib.Nifti1Image(brain_mask.astype(np.int32), dwi_affine), brain_mask_path)

    # Apply bias field correction
    bias_corrected_path = os.path.join(output_dir, "dwi_bias_corrected.nii.gz")
    bias_field_correction(brain_masked_path, bias_corrected_path)

    # Load the bias-corrected DWI data
    bias_corrected = nib.load(bias_corrected_path)

    # Fit the diffusion tensor model
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(bias_corrected.get_fdata())

    # Extract Fractional Anisotropy (FA) and Mean Diffusivity (MD)
    fa_map = tensor_fit.fa
    md_map = tensor_fit.md

    # Save FA and MD maps
    fa_map_path = os.path.join(output_dir, "fa_map.nii.gz")
    md_map_path = os.path.join(output_dir, "md_map.nii.gz")
    nib.save(nib.Nifti1Image(fa_map, dwi_affine), fa_map_path)
    nib.save(nib.Nifti1Image(md_map, dwi_affine), md_map_path)

# Example usage
process_dwi(
    input_dwi="sub-HC131_ses-01_acq-b300-11_dir-AP_dwi.nii.gz",
    bval_file="sub-HC131_ses-01_acq-b300-11_dir-AP_dwi.bval",
    bvec_file="sub-HC131_ses-01_acq-b300-11_dir-AP_dwi.bvec",
    output_dir="output_directory"
)