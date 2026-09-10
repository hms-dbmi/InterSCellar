from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ..core.compute_interscellar_volumes_3d_adaptive import (
        _ARCHIVE_GROUP,
        _load_label_volume,
        _open_label_volume_lazy,
        _open_zarr_store,
        decode_component_mask,
        read_pair,
    )
except ImportError:
    # Invoked as a plain file path rather than as a package module. Load the sibling core
    # module straight off disk instead of importing the package, whose __init__ pulls in
    # napari, cv2 and anndata that this viewer does not need.
    import importlib.util as _importlib_util

    _core_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "core", "compute_interscellar_volumes_3d_adaptive.py",
    )
    _spec = _importlib_util.spec_from_file_location("_interscellar_adaptive_core", _core_path)
    _core = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_core)

    _ARCHIVE_GROUP = _core._ARCHIVE_GROUP
    _load_label_volume = _core._load_label_volume
    _open_label_volume_lazy = _core._open_label_volume_lazy
    _open_zarr_store = _core._open_zarr_store
    decode_component_mask = _core.decode_component_mask
    read_pair = _core.read_pair

DEFAULT_VOXEL_SIZE_UM = (0.56, 0.28, 0.28)

COMPONENT_STYLE = (
    ("territory_a", "territory_A (intracellular)", "royalblue"),
    ("corridor", "corridor (extracellular)", "springgreen"),
    ("territory_b", "territory_B (intracellular)", "mediumorchid"),
)
INTERFACE_STYLE = (
    ("interface_a", "interface_A", "orange"),
    ("interface_b", "interface_B", "gold"),
)


def _archive_group(store):
    if _ARCHIVE_GROUP not in store:
        raise SystemExit(
            f"This store has no '{_ARCHIVE_GROUP}' group, so it predates the lossless "
            f"archive and cannot be read without arbitration loss. Either recompute it "
            f"with compute_interscellar_volumes_3d_adaptive.py, or view it with the older "
            f"visualize_pair_3d, which reads the arbitrated 'interscellar_meshes' preview."
        )
    return store[_ARCHIVE_GROUP]


def _committed_rows(group) -> int:
    return int(group.attrs.get("committed_rows", group["pair_id"].shape[0]))


def _voxel_size(store, override) -> Tuple[float, float, float]:
    if override is not None:
        return tuple(float(v) for v in override)
    stored = store.attrs.get("voxel_size_um") if hasattr(store, "attrs") else None
    try:
        return tuple(float(v) for v in stored)
    except (TypeError, ValueError):
        print(f"  Warning: no voxel_size_um in the store; using {DEFAULT_VOXEL_SIZE_UM}")
        return DEFAULT_VOXEL_SIZE_UM


def _padded_window(origin, shape, volume_shape, pad: int):
    lo = [max(0, int(origin[a]) - pad) for a in range(3)]
    hi = [min(int(volume_shape[a]), int(origin[a] + shape[a]) + pad) for a in range(3)]
    return tuple(slice(lo[a], hi[a]) for a in range(3))


def _place(local: np.ndarray, origin, window) -> np.ndarray:
    out = np.zeros(tuple(s.stop - s.start for s in window), dtype=bool)
    src, dst = [], []
    for axis in range(3):
        start = int(origin[axis])
        lo = max(start, window[axis].start)
        hi = min(start + local.shape[axis], window[axis].stop)
        if hi <= lo:
            return out
        src.append(slice(lo - start, hi - start))
        dst.append(slice(lo - window[axis].start, hi - window[axis].start))
    out[tuple(dst)] = local[tuple(src)]
    return out


def _rows_overlapping(group, n_rows: int, pair_id: int, window):
    origins = np.asarray(group["origin_zyx"][:n_rows]).astype(np.int64)
    shapes = np.asarray(group["shape_zyx"][:n_rows]).astype(np.int64)
    ids = np.asarray(group["pair_id"][:n_rows]).astype(np.int64)

    lo = np.array([w.start for w in window], dtype=np.int64)
    hi = np.array([w.stop for w in window], dtype=np.int64)
    intersects = np.all((origins < hi) & (origins + shapes > lo), axis=1)
    intersects &= ids != int(pair_id)
    return ids[intersects].tolist(), np.flatnonzero(intersects).tolist()


def _decode_footprint(group, row: int) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
    shape = tuple(int(v) for v in np.asarray(group["shape_zyx"][row]))
    origin = tuple(int(v) for v in np.asarray(group["origin_zyx"][row]))
    mask = np.zeros(shape, dtype=bool)
    for name in ("territory_a", "corridor", "territory_b"):
        lo, hi = (int(v) for v in np.asarray(group[f"{name}_indptr"][row:row + 2]))
        if hi > lo:
            mask |= decode_component_mask(np.asarray(group[name][lo:hi]), shape)
    return mask, origin, shape


def load_pair_view(
    interscellar_zarr: str,
    pair_id: int,
    mask_path: Optional[str] = None,
    pad: int = 8,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
    show_overlapping: bool = False,
    show_other_cells: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    store = _open_zarr_store(interscellar_zarr, "interscellar zarr")
    group = _archive_group(store)
    n_rows = _committed_rows(group)
    voxel = _voxel_size(store, voxel_size_um)

    try:
        pair = read_pair(store, int(pair_id))
    except KeyError as exc:
        raise SystemExit(
            f"{exc}\n"
            f"  The archive holds {n_rows} pairs. A pair missing from it was either never "
            f"computed, or rejected because no connected extracellular corridor reached "
            f"both cells -- look for it in the run's *_rejected_pairs.csv."
        ) from exc

    if "interscellar_meshes" in store:
        volume_shape = tuple(int(v) for v in store["interscellar_meshes"].shape[-3:])
    else:
        volume_shape = tuple(
            int(pair["origin_zyx"][a] + pair["shape_zyx"][a]) for a in range(3)
        )

    window = _padded_window(pair["origin_zyx"], pair["shape_zyx"], volume_shape, pad)
    origin = pair["origin_zyx"]

    layers: List[Tuple[str, np.ndarray, str]] = []
    for key, name, colour in COMPONENT_STYLE:
        layers.append((name, _place(pair[key], origin, window), colour))

    interface_layers: List[Tuple[str, np.ndarray, str]] = []
    kind = pair["interface_kind_name"]
    for key, name, colour in INTERFACE_STYLE:
        label = f"{name} ({kind} contact)" if kind == "direct" else f"{name} (facing patch)"
        interface_layers.append((label, _place(pair[key], origin, window), colour))

    # Cell-only: each cell minus its own territory for THIS pair
    cell_layers: List[Tuple[str, np.ndarray, str]] = []
    other_cells = None
    if mask_path:
        volume = _open_label_volume_lazy(mask_path)
        if volume is None:
            volume = _load_label_volume(mask_path)
        if tuple(volume.shape) != volume_shape:
            raise SystemExit(
                f"--mask has shape {tuple(volume.shape)} but the interscellar store is "
                f"{volume_shape}. The segmentation must be the one the volumes were "
                f"computed on."
            )
        crop = np.asarray(volume[window])
        a_id, b_id = pair["cell_a_id"], pair["cell_b_id"]
        terr_a = _place(pair["territory_a"], origin, window)
        terr_b = _place(pair["territory_b"], origin, window)
        cell_layers.append((f"cell_{a_id}_only", (crop == a_id) & ~terr_a, "cornflowerblue"))
        cell_layers.append((f"cell_{b_id}_only", (crop == b_id) & ~terr_b, "plum"))
        if show_other_cells:
            other_cells = np.where((crop != 0) & (crop != a_id) & (crop != b_id), crop, 0)
    elif verbose:
        print("  No --mask given, so cell-only layers are skipped.")

    overlap_layers: List[Tuple[str, np.ndarray, str]] = []
    sharers: List[Tuple[int, int]] = []
    if show_overlapping:
        this = _place(pair["mask"], origin, window)
        candidate_ids, candidate_rows = _rows_overlapping(group, n_rows, pair_id, window)
        shared_any = np.zeros_like(this)
        for other_id, row in zip(candidate_ids, candidate_rows):
            mask_o, origin_o, _ = _decode_footprint(group, row)
            placed = _place(mask_o, origin_o, window)
            common = placed & this
            n = int(common.sum())
            if n:
                sharers.append((int(other_id), n))
                shared_any |= common
        sharers.sort(key=lambda t: -t[1])
        if shared_any.any():
            overlap_layers.append(
                ("shared with other pairs", shared_any, "red")
            )

    # The default view is the whole footprint as one layer, matching visualize_pair_3d.
    # The per-component split stays available behind --split-components for when the
    # intracellular / extracellular breakdown is the question.
    whole = _place(pair["mask"], origin, window)
    volume_layer = (f"interscellar_pair_{pair['pair_id']}", whole, "springgreen")

    return {
        "pair": pair,
        "window": window,
        "voxel_size_um": voxel,
        "volume_shape": volume_shape,
        "volume_layer": volume_layer,
        "component_layers": layers,
        "interface_layers": interface_layers,
        "cell_layers": cell_layers,
        "overlap_layers": overlap_layers,
        "other_cells": other_cells,
        "sharers": sharers,
        "n_rows": n_rows,
    }


def _report(view: Dict[str, Any], split: bool = False) -> None:
    pair, window = view["pair"], view["window"]
    vz, vy, vx = view["voxel_size_um"]
    print(
        f"\nPair {pair['pair_id']}: cells {pair['cell_a_id']} / {pair['cell_b_id']}  "
        f"[{pair['interface_kind_name']}]"
    )
    print(
        f"  {pair['n_voxels']} voxels = {pair['volume_um3']:.2f} um^3   "
        f"territory_A {int(pair['territory_a'].sum())} | corridor {int(pair['corridor'].sum())} "
        f"| territory_B {int(pair['territory_b'].sum())}"
    )
    print(
        f"  interface_A {int(pair['interface_a'].sum())} | interface_B "
        f"{int(pair['interface_b'].sum())}   surface separation "
        f"{pair['min_surface_separation_um']:.3f} um"
    )
    print(
        f"  {pair['n_components']} connected component(s)   "
        f"{pair['shared_voxels']} voxels ({100 * pair['shared_voxels'] / max(1, pair['n_voxels']):.1f}%) "
        f"also claimed by another pair"
    )
    print(
        f"  stored bbox origin {tuple(pair['origin_zyx'])} shape {tuple(pair['shape_zyx'])};"
        f" viewing window z[{window[0].start}:{window[0].stop}] "
        f"y[{window[1].start}:{window[1].stop}] x[{window[2].start}:{window[2].stop}]"
    )
    print(f"  voxel size (z, y, x) um: {(vz, vy, vx)}")
    if view["sharers"]:
        listed = ", ".join(f"{pid} ({n} vox)" for pid, n in view["sharers"][:8])
        more = "" if len(view["sharers"]) <= 8 else f", +{len(view['sharers']) - 8} more"
        print(f"  shares voxels with pair(s): {listed}{more}")

    print("\n  layers, top of the napari list downwards:")
    rows = []
    if split:
        for group in ("interface_layers", "component_layers"):
            rows += [(n, d, c, "") for n, d, c in reversed(view[group])]
    else:
        rows.append((*view["volume_layer"], ""))
    rows += [(n, d, c, "") for n, d, c in reversed(view["cell_layers"])]
    rows += [(n, d, c, "  (starts hidden)") for n, d, c in view["overlap_layers"]]
    for name, data, colour, note in rows:
        n = int(data.sum())
        if n:
            print(f"    {colour:16s} {name:34s} {n:>9,} vox{note}")
    if view["other_cells"] is not None:
        n = int((view["other_cells"] > 0).sum())
        print(f"    {'label colours':16s} {'other cells in crop':34s} {n:>9,} vox"
              f"  (starts hidden)")


def _export(view: Dict[str, Any], path: str) -> None:
    pair, window = view["pair"], view["window"]
    arrays = {
        "window_origin_zyx": np.array([w.start for w in window]),
        "voxel_size_um": np.asarray(view["voxel_size_um"]),
        "pair_id": np.asarray(pair["pair_id"]),
        "cell_a_id": np.asarray(pair["cell_a_id"]),
        "cell_b_id": np.asarray(pair["cell_b_id"]),
        "interface_kind": np.asarray(pair["interface_kind"]),
    }
    for group_key in ("component_layers", "interface_layers", "cell_layers", "overlap_layers"):
        for name, data, _colour in view[group_key]:
            arrays[name.replace(" ", "_")] = data
    if view["other_cells"] is not None:
        arrays["other_cells"] = view["other_cells"]
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"\nWrote {len(arrays)} arrays to {path}")


def _view_in_napari(view: Dict[str, Any], args) -> None:
    try:
        import napari
    except Exception:
        print("Error: napari import failed. Install with: pip install 'napari[all]'")
        raise

    pair = view["pair"]
    scale = view["voxel_size_um"]
    viewer = napari.Viewer(
        title=(
            f"Pair {pair['pair_id']}: cells {pair['cell_a_id']} / {pair['cell_b_id']} "
            f"({pair['interface_kind_name']})"
        ),
        ndisplay=3 if args.view_3d else 2,
    )

    def add(name, data, colour, opacity, visible=True):
        if not data.any():
            return
        # iso rather than attenuated_mip: these are binary masks, so a surface at 0.5 gives
        # a crisp boundary, where a maximum-intensity projection smears every layer into a
        # translucent cloud and nine of those clouds are unreadable.
        viewer.add_image(
            data.astype(np.float32), name=name, colormap=colour, blending="translucent",
            opacity=opacity, rendering="iso", iso_threshold=0.5, scale=scale,
            visible=visible,
        )

    # Cells first and faint: each is typically several times the size of the volume between
    # them, so at equal opacity they bury it. Keeping them translucent is what lets the
    # volume's intracellular reach read through the cell it sits inside.
    if view["other_cells"] is not None:
        viewer.add_labels(
            view["other_cells"].astype(np.uint32), name="other cells in crop",
            opacity=0.25, scale=scale, visible=False,
        )
    for name, data, colour in view["cell_layers"]:
        add(name, data, colour, args.cells_opacity)

    if args.split_components:
        for name, data, colour in view["component_layers"]:
            add(name, data, colour, args.pair_opacity)
        for name, data, colour in view["interface_layers"]:
            add(name, data, colour, args.interface_opacity)
    else:
        name, data, colour = view["volume_layer"]
        add(name, data, colour, args.pair_opacity)

    for name, data, colour in view["overlap_layers"]:
        add(name, data, colour, 0.85, visible=False)

    # Open on the pair's busiest slice rather than slice 0, which is usually empty.
    if not args.view_3d:
        mask = view["volume_layer"][1]
        per_slice = mask.reshape(mask.shape[0], -1).sum(axis=1)
        viewer.dims.set_current_step(0, int(np.argmax(per_slice)))

    napari.run()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one interscellar pair from the adaptive pipeline's lossless pair "
            "archive. Every voxel the pair claims is shown, including voxels other pairs "
            "also claim, which the dense 'interscellar_meshes' preview cannot represent."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interscellar-zarr", required=True,
        help="Adaptive interscellar volumes zarr containing the 'pair_volumes' group.",
    )
    parser.add_argument("--pair-id", type=int, help="Pair to show.")
    parser.add_argument(
        "--mask", default=None,
        help="Segmentation the volumes were computed on. Needed for the cell-only layers; "
             "read lazily, so only the pair's own crop is loaded.",
    )
    parser.add_argument(
        "--bbox-pad", type=int, default=8,
        help="Context voxels around the pair's stored bounding box.",
    )
    parser.add_argument(
        "--show-overlapping", action="store_true",
        help="Add a layer marking voxels other pairs also claim, and list those pairs.",
    )
    parser.add_argument(
        "--show-other-cells", action="store_true",
        help="Add a faint layer of all other cell labels in the crop.",
    )
    parser.add_argument(
        "--voxel-size-um", nargs=3, type=float, default=None, metavar=("Z", "Y", "X"),
        help="Override the store's voxel_size_um, used to scale the 3D view.",
    )
    parser.add_argument(
        "--3d", dest="view_3d", action="store_true",
        help="Open in 3D volume mode. The default 2D slice view is far easier to read, "
             "since a pair's cells are usually several times the size of the volume "
             "between them and hide it when everything is rendered translucently.",
    )
    parser.add_argument(
        "--split-components", action="store_true",
        help="Show territory_A / corridor / territory_B and both interfaces as separate "
             "layers instead of the interscellar volume as one.",
    )
    parser.add_argument(
        "--pair-opacity", type=float, default=0.9,
        help="Opacity of the interscellar volume.",
    )
    parser.add_argument(
        "--cells-opacity", type=float, default=0.25,
        help="Opacity of the two cell-only volumes. Low on purpose, so the volume's "
             "intracellular reach shows through the cell around it.",
    )
    parser.add_argument(
        "--interface-opacity", type=float, default=1.0,
        help="Opacity of the interface layers, with --split-components.",
    )
    parser.add_argument(
        "--export-crop", default=None,
        help="Save every layer to this .npz instead of only viewing it.",
    )
    parser.add_argument(
        "--no-view", action="store_true",
        help="Print the summary (and export, if asked) without opening napari.",
    )
    parser.add_argument(
        "--list", type=int, default=0, metavar="N",
        help="List the N pairs with the largest volume in this archive, then exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        store = _open_zarr_store(args.interscellar_zarr, "interscellar zarr")
        group = _archive_group(store)
        n_rows = _committed_rows(group)
        ids = np.asarray(group["pair_id"][:n_rows])
        vox = np.asarray(group["n_voxels"][:n_rows])
        kind = np.asarray(group["interface_kind"][:n_rows])
        shared = np.asarray(group["shared_voxels"][:n_rows])
        order = np.argsort(vox)[::-1][:args.list]
        print(f"{n_rows} pairs in {args.interscellar_zarr}")
        print(f"{'pair_id':>9} {'cells':>16} {'voxels':>8} {'kind':>7} {'shared':>8} {'%shared':>8}")
        a = np.asarray(group["cell_a_id"][:n_rows])
        b = np.asarray(group["cell_b_id"][:n_rows])
        for i in order:
            pct = 100.0 * shared[i] / max(1, vox[i])
            print(
                f"{int(ids[i]):>9} {f'{int(a[i])}/{int(b[i])}':>16} {int(vox[i]):>8} "
                f"{('direct' if kind[i] == 1 else 'near'):>7} {int(shared[i]):>8} {pct:>7.1f}%"
            )
        return

    if args.pair_id is None:
        parser.error("--pair-id is required (or use --list N to see what is available)")

    view = load_pair_view(
        interscellar_zarr=args.interscellar_zarr,
        pair_id=args.pair_id,
        mask_path=args.mask,
        pad=args.bbox_pad,
        voxel_size_um=tuple(args.voxel_size_um) if args.voxel_size_um else None,
        show_overlapping=args.show_overlapping,
        show_other_cells=args.show_other_cells,
    )
    _report(view, split=args.split_components)

    if args.export_crop:
        _export(view, args.export_crop)
    if not args.no_view:
        _view_in_napari(view, args)


if __name__ == "__main__":
    main()
