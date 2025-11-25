import argparse
import sys
import os
from pathlib import Path
import numpy as np

try:
    import zarr
except ImportError:
    print("Error: zarr not installed. Install with: pip install zarr")
    sys.exit(1)

try:
    import napari
except ImportError:
    print("Error: napari not installed. Install with: pip install 'napari[all]'")
    sys.exit(1)

def _find_file(filename: str, script_dir: str) -> str:
    possible_paths = [
        filename,
        os.path.join(script_dir, filename),
        os.path.join(os.path.dirname(script_dir), filename),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return filename

def main():
    parser = argparse.ArgumentParser(
        description="Visualize full cell-only and interscellar volumes in Napari"
    )
    parser.add_argument(
        '--cell-only-zarr',
        type=str,
        required=True,
        help='Path to cell-only volumes zarr file'
    )
    parser.add_argument(
        '--interscellar-zarr',
        type=str,
        required=True,
        help='Path to interscellar volumes zarr file'
    )
    parser.add_argument(
        '--cell-only-opacity',
        type=float,
        default=0.7,
        help='Opacity for cell-only volumes layer (0.0-1.0)'
    )
    parser.add_argument(
        '--interscellar-opacity',
        type=float,
        default=0.9,
        help='Opacity for interscellar volumes layer (0.0-1.0)'
    )
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    cell_only_path = _find_file(args.cell_only_zarr, script_dir)
    interscellar_path = _find_file(args.interscellar_zarr, script_dir)
    
    if not os.path.exists(cell_only_path):
        print(f"Error: Cell-only zarr file not found: {cell_only_path}")
        sys.exit(1)
    
    if not os.path.exists(interscellar_path):
        print(f"Error: Interscellar zarr file not found: {interscellar_path}")
        sys.exit(1)
    
    print("Loading zarr files...")
    print(f"Cell-only zarr: {cell_only_path}")
    print(f"Interscellar zarr: {interscellar_path}")
    
    cell_only_zarr = zarr.open(cell_only_path, mode='r')
    
    if 'labels' in cell_only_zarr:
        cell_only_labels = cell_only_zarr['labels']
    elif '0' in cell_only_zarr:
        if isinstance(cell_only_zarr['0'], zarr.hierarchy.Group):
            if '0' in cell_only_zarr['0']:
                cell_only_labels = cell_only_zarr['0']['0']
            else:
                print(f"Error: Unexpected zarr structure in cell-only zarr")
                sys.exit(1)
        else:
            cell_only_labels = cell_only_zarr['0']
    else:
        found = False
        for key in cell_only_zarr.keys():
            node = cell_only_zarr[key]
            if hasattr(node, 'ndim') and node.ndim >= 3:
                cell_only_labels = node
                found = True
                print(f"  Found data in key '{key}'")
                break
        if not found:
            print(f"Error: Could not find data in cell-only zarr")
            print(f"Available keys: {list(cell_only_zarr.keys())}")
            sys.exit(1)
    
    if cell_only_labels.ndim == 5:
        print(f"Cell-only zarr shape (5D): {cell_only_labels.shape}")
        print(f"Using [0, 0, ...] for 3D visualization")
        cell_only_3d = cell_only_labels[0, 0]
    else:
        cell_only_3d = cell_only_labels
    
    print(f"Cell-only volumes shape: {cell_only_3d.shape}")
    
    interscellar_zarr = zarr.open(interscellar_path, mode='r')
    
    if 'interscellar_meshes' in interscellar_zarr:
        interscellar_labels = interscellar_zarr['interscellar_meshes']
    else:
        print(f"Error: Could not find 'interscellar_meshes' in interscellar zarr")
        sys.exit(1)
    
    print(f"  Interscellar volumes shape: {interscellar_labels.shape}")
    
    if cell_only_3d.shape != interscellar_labels.shape:
        print(f"Warning: Shape mismatch!")
        print(f"Cell-only: {cell_only_3d.shape}")
        print(f"Interscellar: {interscellar_labels.shape}")
        print(f"This may cause alignment issues")
    else:
        print(f"Shapes match: {cell_only_3d.shape}")
    
    cell_only_attrs = dict(cell_only_zarr.attrs) if hasattr(cell_only_zarr, 'attrs') else {}
    interscellar_attrs = dict(interscellar_zarr.attrs) if hasattr(interscellar_zarr, 'attrs') else {}
    
    print(f"\nCell-only zarr metadata:")
    for key, value in cell_only_attrs.items():
        print(f"  {key}: {value}")
    
    print(f"\nInterscellar zarr metadata:")
    for key, value in interscellar_attrs.items():
        print(f"  {key}: {value}")
    
    print(f"\nLaunching Napari viewer...")
    
    viewer = napari.Viewer(title="Full Volumes Visualization")
    
    print(f"  Adding cell-only volumes layer...")
    viewer.add_labels(
        cell_only_3d,
        name="cell_only_volumes",
        opacity=args.cell_only_opacity
    )
    
    print(f"  Adding interscellar volumes layer...")
    viewer.add_labels(
        interscellar_labels,
        name="interscellar_volumes",
        opacity=args.interscellar_opacity
    )
    
    if cell_only_3d.shape:
        viewer.camera.center = (
            cell_only_3d.shape[2] / 2,
            cell_only_3d.shape[1] / 2,
        )
        viewer.camera.zoom = 0.5
    
    print(f"\nViewer launched successfully!")
    print(f"Use layer visibility to toggle between cell-only and interscellar volumes")
    print(f"Adjust opacity sliders to blend the layers")
    
    napari.run()

if __name__ == "__main__":
    main()
