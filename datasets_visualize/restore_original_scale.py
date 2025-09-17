# Tool for OReX `.stl` and `.obj` files scale restoration

import numpy as np
import trimesh
import os
import pathlib
import sys
root_path = str(pathlib.Path(__file__).absolute().parent.parent)
sys.path.append(root_path)


def calculate_scale_factor(gt_file, normalized_file):
    """Calculate scale factor between ground truth and normalized files."""
    print(f"=== CALCULATING SCALE FACTOR ===")
    print(f"Ground Truth: {gt_file}")
    print(f"Normalized Reference: {normalized_file}")
    
    # Load files
    gt_mesh = trimesh.load_mesh(gt_file)
    norm_mesh = trimesh.load_mesh(normalized_file)
    
    # Calculate ranges
    gt_range = np.max(gt_mesh.vertices, axis=0) - np.min(gt_mesh.vertices, axis=0)
    norm_range = np.max(norm_mesh.vertices, axis=0) - np.min(norm_mesh.vertices, axis=0)
    
    # Calculate scale factors
    scale_factors = gt_range / norm_range
    avg_scale = np.mean(scale_factors)
    
    print(f"Ground truth range: {gt_range}")
    print(f"Normalized range: {norm_range}")
    print(f"Scale factors per axis: {scale_factors}")
    print(f"Average scale factor: {avg_scale:.6f}")
    
    # Check if scaling is uniform (should be for OReX)
    scale_std = np.std(scale_factors)
    if scale_std < 0.01:
        print("✅ Uniform scaling detected")
        return avg_scale
    else:
        print(f"⚠️  Non-uniform scaling detected (std: {scale_std:.6f})")
        return scale_factors


def scale_back_mesh(input_file, output_file, scale_factor):
    """Scale back a mesh file."""
    print(f"\nScaling: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
    
    # Load mesh
    mesh = trimesh.load_mesh(input_file)
    original_range = np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0)
    
    # Scale around centroid to preserve relative positioning
    centroid = np.mean(mesh.vertices, axis=0)
    centered_verts = mesh.vertices - centroid
    
    # Apply scaling
    if np.isscalar(scale_factor):
        scaled_verts = centered_verts * scale_factor
    else:
        scaled_verts = centered_verts * scale_factor
    
    final_verts = scaled_verts + centroid
    
    # Create scaled mesh
    scaled_mesh = trimesh.Trimesh(vertices=final_verts, faces=mesh.faces)
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    scaled_mesh.export(output_file)
    
    # Verify
    final_range = np.max(final_verts, axis=0) - np.min(final_verts, axis=0)
    print(f"  Original range: {original_range}")
    print(f"  Scaled range: {final_range}")
    print(f"  Scale ratio: {np.mean(final_range / original_range):.6f}")
    print(f"  ✅ Saved to: {output_file}")


def restore_original_scale(gt_file, input_files, input_dir, output_dir):
    # Check ground truth file exists
    if not os.path.exists(gt_file):
        print(f"❌ Ground truth file not found: {gt_file}")
        return
    
    # Check that we have input files
    if not input_files:
        print("❌ No input files specified in input_files list")
        return
    
    # Use the first input file as the normalized reference for scale calculation
    # (assuming the first file is the normalized version of the ground truth)
    if len(input_files) == 0:
        print("❌ Need at least one input file to calculate scale")
        return
        
    normalized_reference = input_files[0]
    if not os.path.exists(normalized_reference):
        print(f"❌ Normalized reference file not found: {normalized_reference}")
        return
    
    # Calculate scale factor
    scale_factor = calculate_scale_factor(gt_file, normalized_reference)
    
    # Process all input files
    print(f"\n=== SCALING {len(input_files)} FILES ===")
    
    for i, input_file in enumerate(input_files, 1):
        if not os.path.exists(input_file):
            print(f"❌ Input file {i} not found: {input_file}")
            continue
        
        # Generate output filename - convert to OBJ format
        filepath_relative = pathlib.Path(input_file).relative_to(input_dir)
        obj_filepath = f"{pathlib.Path(output_dir).joinpath(filepath_relative)}.obj"
        
        try:
            scale_back_mesh(input_file, obj_filepath, scale_factor)
        except Exception as e:
            print(f"❌ Error processing {input_file}: {e}")
    
    print(f"\n🎉 Scaling complete! Used scale factor: {scale_factor}")
    print(f"📁 All results saved to: {output_dir}")


def main():
    # Files to process
    stem_list = [
        "PA000150",
        "PA000157",
        "PA000162",
        "PA000168",
        "PA000169",
    ]

    input_dir = pathlib.Path(root_path).joinpath("datasets_visualize/orex/data_original")
    output_dir = pathlib.Path(root_path).joinpath("datasets_visualize/orex/data_input")
    os.makedirs(output_dir, exist_ok=True)

    for stem in stem_list:
        # First scaling group: parse_labels
        src_file1 = f"{input_dir}/labels/input/{stem}.obj"
        input_files1 = [
            f"{input_dir}/labels/slices/{stem}.stl",
        ]
        
        # Second scaling group: parse_preds_fixed
        src_file2 = f"{input_dir}/preds_fixed/input/{stem}_vessel.obj"
        input_files2 = [
            f"{input_dir}/preds_fixed/slices/{stem}_vessel.stl",
            f"{input_dir}/preds_fixed/output/{stem}_vessel.obj",
        ]
        
        print("🔄 Processing parse_labels files...")
        restore_original_scale(src_file1, input_files1, input_dir, output_dir)
        
        print("\n" + "="*60)
        print("🔄 Processing parse_preds_fixed files...")
        restore_original_scale(src_file2, input_files2, input_dir, output_dir)


if __name__ == "__main__":
    main()
