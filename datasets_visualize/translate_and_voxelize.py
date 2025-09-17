# Tool for Convolutional Occupancy Networks `.off` and `.ply` files alignment

import numpy as np
import open3d as o3d
import trimesh
import pathlib
from typing import List, Tuple, Union
import sys
import os
root_path = str(pathlib.Path(__file__).absolute().parent.parent)
sys.path.append(root_path)

from datasets.dataset_utils import _convert_numpy_to_obj, _convert_numpy_to_npy


def load_mesh_or_pointcloud(filepath: str) -> Union[trimesh.Trimesh, np.ndarray]:
    """Load a mesh or point cloud from various formats"""
    filepath = pathlib.Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = filepath.suffix.lower()
    
    try:
        if ext in ['.off', '.obj']:
            mesh = trimesh.load(str(filepath))
            if hasattr(mesh, 'vertices'):
                return mesh
            else:
                raise ValueError(f"Could not load mesh from {filepath}")
                
        elif ext == '.ply':
            try:
                mesh = trimesh.load(str(filepath))
                if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                    return mesh
                else:
                    pcd = o3d.io.read_point_cloud(str(filepath))
                    if len(pcd.points) == 0:
                        raise ValueError(f"Empty point cloud: {filepath}")
                    return np.asarray(pcd.points)
            except:
                pcd = o3d.io.read_point_cloud(str(filepath))
                if len(pcd.points) == 0:
                    raise ValueError(f"Empty point cloud: {filepath}")
                return np.asarray(pcd.points)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    except Exception as e:
        raise ValueError(f"Failed to load {filepath}: {str(e)}")


def get_vertices(data: Union[trimesh.Trimesh, np.ndarray]) -> np.ndarray:
    """Extract vertices from mesh or point cloud"""
    if isinstance(data, trimesh.Trimesh):
        return data.vertices
    else:
        return data


def calculate_translation_to_positive(files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate translation needed to move all files to positive coordinates.
    
    Returns:
        global_min_bounds, translation_vector, max_translated_bounds
    """
    print("Calculating translation to positive coordinates...")
    
    all_min_bounds = []
    all_max_bounds = []
    
    for filepath in files:
        print(f"  Loading {filepath}...")
        data = load_mesh_or_pointcloud(filepath)
        vertices = get_vertices(data)
        
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        all_min_bounds.append(min_coords)
        all_max_bounds.append(max_coords)
        
        print(f"    Vertices: {len(vertices)}")
        print(f"    Bounds: [{min_coords[0]:.2f}, {min_coords[1]:.2f}, {min_coords[2]:.2f}] to [{max_coords[0]:.2f}, {max_coords[1]:.2f}, {max_coords[2]:.2f}]")
    
    # Find global minimum bounds across all files
    global_min_bounds = np.min(all_min_bounds, axis=0)
    global_max_bounds = np.max(all_max_bounds, axis=0)
    
    # Calculate translation to move global minimum to small positive values (e.g., [1, 1, 1])
    positive_offset = np.array([1.0, 1.0, 1.0])  # Small offset from origin
    translation_vector = positive_offset - global_min_bounds
    
    # Calculate what the max bounds will be after translation
    max_translated_bounds = global_max_bounds + translation_vector
    translated_size = max_translated_bounds - positive_offset
    
    print(f"\nTranslation analysis:")
    print(f"  Global min bounds: [{global_min_bounds[0]:.2f}, {global_min_bounds[1]:.2f}, {global_min_bounds[2]:.2f}]")
    print(f"  Global max bounds: [{global_max_bounds[0]:.2f}, {global_max_bounds[1]:.2f}, {global_max_bounds[2]:.2f}]")
    print(f"  Translation vector: [{translation_vector[0]:.2f}, {translation_vector[1]:.2f}, {translation_vector[2]:.2f}]")
    print(f"  After translation:")
    print(f"    New min bounds: [{positive_offset[0]:.2f}, {positive_offset[1]:.2f}, {positive_offset[2]:.2f}]")
    print(f"    New max bounds: [{max_translated_bounds[0]:.2f}, {max_translated_bounds[1]:.2f}, {max_translated_bounds[2]:.2f}]")
    print(f"    Total size: [{translated_size[0]:.2f}, {translated_size[1]:.2f}, {translated_size[2]:.2f}]")
    
    # Check if it fits in 512³ box
    max_dimension = np.max(translated_size)
    print(f"    Max dimension: {max_dimension:.2f}")
    
    if max_dimension <= 512:
        print(f"  ✅ Fits perfectly in 512³ box!")
    else:
        print(f"  ⚠️ Exceeds 512³ box by {max_dimension - 512:.2f} units")
        print(f"     Consider using {int(np.ceil(max_dimension))}³ box instead")
    
    return global_min_bounds, translation_vector, max_translated_bounds


def voxelize_translated_data(data: Union[trimesh.Trimesh, np.ndarray], 
                           translation_vector: np.ndarray,
                           grid_size: int = 512,
                           voxel_size: float = 1.0) -> np.ndarray:
    """Voxelize data after translating to positive coordinates"""
    
    vertices = get_vertices(data)
    
    # Apply translation
    translated_vertices = vertices + translation_vector
    
    # Create coordinate grids for the 512³ box
    x_edges = np.linspace(0, grid_size * voxel_size, grid_size + 1)
    y_edges = np.linspace(0, grid_size * voxel_size, grid_size + 1)
    z_edges = np.linspace(0, grid_size * voxel_size, grid_size + 1)
    
    # Digitize points into voxel indices
    x_indices = np.digitize(translated_vertices[:, 0], x_edges) - 1
    y_indices = np.digitize(translated_vertices[:, 1], y_edges) - 1
    z_indices = np.digitize(translated_vertices[:, 2], z_edges) - 1
    
    # Clamp indices to valid range
    x_indices = np.clip(x_indices, 0, grid_size - 1)
    y_indices = np.clip(y_indices, 0, grid_size - 1)
    z_indices = np.clip(z_indices, 0, grid_size - 1)
    
    # Create voxel grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    voxel_grid[x_indices, y_indices, z_indices] = 1
    
    return voxel_grid


def process_files_with_translation(files: List[str], input_dir: str, output_dir: str, grid_size: int = 512, voxel_size: float = 1.0, export_obj: bool = False) -> None:
    """
    Process files by translating to positive coordinates and voxelizing in grid_size^3 box.
    """
    print("="*80)
    print(f"TRANSLATION TO POSITIVE + {grid_size}^3 VOXELIZATION")
    print("="*80)
    
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Calculate translation needed
    global_min_bounds, translation_vector, max_translated_bounds = calculate_translation_to_positive(files)

    print(f"\nProcessing {len(files)} files with translation + {grid_size}^3 voxelization...")
    print(f"Output directory: {output_path}")
    
    # Process each file
    results = []
    for i, filepath in enumerate(files):
        print(f"\nProcessing file {i+1}/{len(files)}: {filepath}")
        
        # Load data
        data = load_mesh_or_pointcloud(filepath)
        data_type = "mesh" if isinstance(data, trimesh.Trimesh) else "pointcloud"
        vertices = get_vertices(data)
        
        print(f"  Type: {data_type}")
        print(f"  Vertices: {len(vertices)}")
        
        # Apply translation and check bounds
        translated_vertices = vertices + translation_vector
        new_min = np.min(translated_vertices, axis=0)
        new_max = np.max(translated_vertices, axis=0)
        print(f"  After translation: [{new_min[0]:.2f}, {new_min[1]:.2f}, {new_min[2]:.2f}] to [{new_max[0]:.2f}, {new_max[1]:.2f}, {new_max[2]:.2f}]")
        
        # Voxelize in 512³ box
        print(f"  Voxelizing in {grid_size}³ box...")
        voxel_grid = voxelize_translated_data(data, translation_vector, grid_size, voxel_size)
        volume = np.sum(voxel_grid)
        
        print(f"  Voxel grid shape: {voxel_grid.shape}")
        print(f"  Volume (filled voxels): {volume}")
        
        # Generate output filename
        filepath_relative = pathlib.Path(filepath).relative_to(input_dir)
        npy_filepath = f"{pathlib.Path(output_path).joinpath(filepath_relative)}.npy"
        
        # Save voxel grid
        _convert_numpy_to_npy(numpy_data=voxel_grid, save_filename=npy_filepath)
        print(f"  Saved: {npy_filepath}")
        
        # Convert to OBJ using the imported function from dataset_utils
        if export_obj:
            obj_filepath = str(pathlib.Path(npy_filepath).with_suffix('.obj'))
            print(f"  Converting to OBJ...")
            _convert_numpy_to_obj(voxel_grid, save_filename=obj_filepath, mesh_scale=1.0, voxel_size=1.0)
            print(f"  Saved: {obj_filepath}")

        results.append({
            'input_file': filepath,
            'output_file': npy_filepath,
            'shape': voxel_grid.shape,
            'volume': volume,
            'type': data_type,
            'translated_bounds': (new_min, new_max)
        })
    
    # Summary
    print("\n" + "="*80)
    print("TRANSLATION + VOXELIZATION COMPLETE")
    print("="*80)
    print(f"Grid size: {grid_size}^3")
    print(f"Voxel size: {voxel_size}")
    print(f"Translation applied: [{translation_vector[0]:.2f}, {translation_vector[1]:.2f}, {translation_vector[2]:.2f}]")
    print(f"All files now in positive coordinate space")
    print(f"Output directory: {output_path}")
    print()
    
    for i, result in enumerate(results, 1):
        print(f"File {i}: {pathlib.Path(result['input_file']).name}")
        print(f"  → {pathlib.Path(result['output_file']).name}")
        print(f"  → Shape: {result['shape']}, Volume: {result['volume']}, Type: {result['type']}")

    print(f"\nAll files translated and voxelized in {grid_size}^3 box!")

    # Calculate Dice scores for verification
    # if len(results) >= 2:
    #     print(f"\nDice score verification:")
        
    #     # Load and compare files
    #     for i in range(len(results)):
    #         for j in range(i+1, len(results)):
    #             file1_data = np.load(results[i]['output_file'])
    #             file2_data = np.load(results[j]['output_file'])
                
    #             # Calculate Dice
    #             intersection = np.sum((file1_data > 0) & (file2_data > 0))
    #             total = np.sum(file1_data > 0) + np.sum(file2_data > 0)
    #             dice = 2.0 * intersection / total if total > 0 else 0

    #             file1_name = pathlib.Path(results[i]['input_file']).name
    #             file2_name = pathlib.Path(results[j]['input_file']).name
    #             print(f"  {file1_name} vs {file2_name}: Dice = {dice:.4f}")


def main():
    # Files to process
    stem_list = [
        "PA000150",
        "PA000157",
        "PA000162",
        "PA000168",
        "PA000169",
    ]
    input_dir = pathlib.Path(root_path).joinpath("datasets_visualize/data_input")
    output_dir = pathlib.Path(root_path).joinpath("datasets_visualize/data_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process with translation to positive coordinates + 512^3 voxelization
    grid_size = 512
    voxel_size = 1.0
    export_obj = True

    for stem in stem_list:
        print(f"\nProcessing stem: {stem}")
        filepaths = sorted(input_dir.rglob(f"{stem}*.*"))
        if not filepaths:
            print(f"  No files found for stem {stem} in {input_dir}")
            continue
        
        process_files_with_translation(
            filepaths, 
            input_dir, 
            output_dir, 
            grid_size, 
            voxel_size, 
            export_obj=export_obj
        )


if __name__ == "__main__":
    main()