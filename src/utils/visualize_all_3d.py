import argparse
import sys
import os
from pathlib import Path
import numpy as np

try:
    import zarr
    try:
        from zarr.hierarchy import Group as ZarrGroup
    except ImportError:
        ZarrGroup = zarr.Group
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


def _as_3d_labels(arr):
    if arr.ndim == 5:
        return arr[0, 0]
    if arr.ndim == 4:
        return arr[0]
    return arr


def _interscellar_3d_view(zroot):
    if 'interscellar_meshes' in zroot:
        arr = zroot['interscellar_meshes']
    elif '0' in zroot:
        if isinstance(zroot['0'], ZarrGroup) and '0' in zroot['0']:
            arr = zroot['0']['0']
        else:
            arr = zroot['0']
    else:
        found = False
        for key in zroot.keys():
            node = zroot[key]
            if hasattr(node, 'ndim') and node.ndim >= 3:
                arr = node
                found = True
                print(f"  Found interscellar data in key '{key}'")
                break
        if not found:
            print("Error: Could not find interscellar volume data in zarr")
            print(f"Available keys: {list(zroot.keys())}")
            sys.exit(1)

    if arr.ndim == 5:
        print(f"Interscellar zarr shape (5D): {arr.shape}")
        print("Using [0, 0, ...] for 3D visualization")
        return arr[0, 0]
    return arr


def _cell_only_3d_view(zroot):
    if 'labels' in zroot:
        cell_only_labels = zroot['labels']
    elif '0' in zroot:
        if isinstance(zroot['0'], ZarrGroup):
            if '0' in zroot['0']:
                cell_only_labels = zroot['0']['0']
            else:
                print(f"Error: Unexpected zarr structure in cell-only zarr")
                sys.exit(1)
        else:
            cell_only_labels = zroot['0']
    else:
        found = False
        for key in zroot.keys():
            node = zroot[key]
            if hasattr(node, 'ndim') and node.ndim >= 3:
                cell_only_labels = node
                found = True
                print(f"  Found data in key '{key}'")
                break
        if not found:
            print(f"Error: Could not find data in cell-only zarr")
            print(f"Available keys: {list(zroot.keys())}")
            sys.exit(1)

    if cell_only_labels.ndim == 5:
        print(f"Cell-only zarr shape (5D): {cell_only_labels.shape}")
        print(f"Using [0, 0, ...] for 3D visualization")
        return cell_only_labels[0, 0]
    return cell_only_labels


def _combined_3d_view(zroot):
    if '0' in zroot:
        node = zroot['0']
        if isinstance(node, ZarrGroup) and '0' in node:
            arr = node['0']
        else:
            arr = node
    elif 'labels' in zroot:
        arr = zroot['labels']
    else:
        found = False
        for key in zroot.keys():
            node = zroot[key]
            if hasattr(node, 'ndim') and node.ndim >= 3:
                arr = node
                found = True
                print(f"  Found combined data in key '{key}'")
                break
        if not found:
            print("Error: Could not find combined volume data in zarr")
            print(f"Available keys: {list(zroot.keys())}")
            sys.exit(1)

    if arr.ndim == 5:
        print(f"Combined zarr shape (5D): {arr.shape}")
        print("Using [0, 0, ...] for 3D visualization")
        return arr[0, 0]
    return _as_3d_labels(arr)


def _print_attrs(title: str, zroot) -> None:
    attrs = dict(zroot.attrs) if hasattr(zroot, 'attrs') else {}
    print(f"\n{title}:")
    if not attrs:
        print("  (none)")
        return
    for key, value in attrs.items():
        print(f"  {key}: {value}")


def _resolve_existing_zarr(path_arg: str, script_dir: str, label: str):
    resolved = _find_file(path_arg, script_dir)
    if not os.path.exists(resolved):
        print(f"Error: {label} zarr file not found: {resolved}")
        sys.exit(1)
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize cell-only, interscellar, and/or combined volumes in Napari. "
            "Provide any one file, or both cell-only and interscellar as before."
        )
    )
    parser.add_argument(
        '--cell-only-zarr',
        type=str,
        default=None,
        help='Path to cell-only volumes zarr file'
    )
    parser.add_argument(
        '--interscellar-zarr',
        type=str,
        default=None,
        help='Path to interscellar volumes zarr file'
    )
    parser.add_argument(
        '--combined-zarr',
        type=str,
        default=None,
        help='Path to combined volumes zarr from combine_volumes_3d'
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
    parser.add_argument(
        '--combined-opacity',
        type=float,
        default=0.8,
        help='Opacity for combined volumes layer (0.0-1.0)'
    )
    
    args = parser.parse_args()
    
    if not args.cell_only_zarr and not args.interscellar_zarr and not args.combined_zarr:
        parser.error(
            "provide at least one of --cell-only-zarr, --interscellar-zarr, or --combined-zarr"
        )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    layers = []
    reference_shape = None
    cell_only_3d = None
    interscellar_labels = None

    print("Loading zarr files...")

    if args.cell_only_zarr:
        cell_only_path = _resolve_existing_zarr(
            args.cell_only_zarr, script_dir, "Cell-only"
        )
        print(f"Cell-only zarr: {cell_only_path}")
        cell_only_zarr = zarr.open(cell_only_path, mode='r')
        cell_only_3d = _cell_only_3d_view(cell_only_zarr)
        print(f"Cell-only volumes shape: {cell_only_3d.shape}")
        _print_attrs("Cell-only zarr metadata", cell_only_zarr)
        layers.append((cell_only_3d, "cell_only_volumes", args.cell_only_opacity))
        reference_shape = cell_only_3d.shape

    if args.interscellar_zarr:
        interscellar_path = _resolve_existing_zarr(
            args.interscellar_zarr, script_dir, "Interscellar"
        )
        print(f"Interscellar zarr: {interscellar_path}")
        interscellar_zarr = zarr.open(interscellar_path, mode='r')
        interscellar_labels = _interscellar_3d_view(interscellar_zarr)
        print(f"  Interscellar volumes shape: {interscellar_labels.shape}")
        _print_attrs("Interscellar zarr metadata", interscellar_zarr)
        layers.append(
            (interscellar_labels, "interscellar_volumes", args.interscellar_opacity)
        )
        if reference_shape is None:
            reference_shape = interscellar_labels.shape

    if args.combined_zarr:
        combined_path = _resolve_existing_zarr(
            args.combined_zarr, script_dir, "Combined"
        )
        print(f"Combined zarr: {combined_path}")
        combined_zarr = zarr.open(combined_path, mode='r')
        combined_3d = _combined_3d_view(combined_zarr)
        print(f"Combined volumes shape: {combined_3d.shape}")
        _print_attrs("Combined zarr metadata", combined_zarr)
        max_cell_id = combined_zarr.attrs.get('max_cell_only_id')
        n_cell_ids = combined_zarr.attrs.get('n_cell_only_ids')
        n_inter_ids = combined_zarr.attrs.get('n_interscellar_ids')
        if max_cell_id is not None:
            print(
                f"\nID ranges: cell-only 1..{int(max_cell_id)}; "
                f"interscellar starts at {int(max_cell_id) + 1}"
            )
        if n_cell_ids is not None and n_inter_ids is not None:
            print(
                f"Unique IDs: {int(n_cell_ids)} cell-only, {int(n_inter_ids)} interscellar"
            )
        layers.append((combined_3d, "combined_volumes", args.combined_opacity))
        if reference_shape is None:
            reference_shape = combined_3d.shape

    if cell_only_3d is not None and interscellar_labels is not None:
        if cell_only_3d.shape != interscellar_labels.shape:
            print(f"Warning: Shape mismatch!")
            print(f"Cell-only: {cell_only_3d.shape}")
            print(f"Interscellar: {interscellar_labels.shape}")
            print(f"This may cause alignment issues")
        else:
            print(f"Shapes match: {cell_only_3d.shape}")
    
    print(f"\nLaunching Napari viewer...")
    
    viewer = napari.Viewer(title="Full Volumes Visualization")
    for data, name, opacity in layers:
        print(f"  Adding {name} layer...")
        viewer.add_labels(data, name=name, opacity=opacity)
    
    if reference_shape:
        viewer.camera.center = (
            reference_shape[2] / 2,
            reference_shape[1] / 2,
        )
        viewer.camera.zoom = 0.5
    
    print(f"\nViewer launched successfully!")
    if len(layers) > 1:
        print(f"Use layer visibility to toggle between loaded volumes")
        print(f"Adjust opacity sliders to blend the layers")
    else:
        print(f"Showing {layers[0][1]} in Napari")
    
    napari.run()

if __name__ == "__main__":
    main()
