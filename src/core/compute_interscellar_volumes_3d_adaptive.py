# Import

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

_CONN_26 = generate_binary_structure(3, 3)

def _pair_union_bbox(
    halo_bboxes: Dict[int, Tuple[slice, slice, slice]],
    cell_a_id: int,
    cell_b_id: int,
) -> Tuple[slice, slice, slice]:
    if cell_a_id not in halo_bboxes or cell_b_id not in halo_bboxes:
        raise KeyError(f"Missing halo bbox for one of the pair cells: {(cell_a_id, cell_b_id)}")

    bbox_a = halo_bboxes[cell_a_id]
    bbox_b = halo_bboxes[cell_b_id]

    return tuple(
        slice(min(bbox_a[axis].start, bbox_b[axis].start), max(bbox_a[axis].stop, bbox_b[axis].stop))
        for axis in range(3)
    )


def _fallback_union_bbox(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    max_distance_um: float,
) -> Tuple[slice, slice, slice]:
    coords = np.concatenate([np.argwhere(mask_a), np.argwhere(mask_b)], axis=0)
    lower = coords.min(axis=0)
    upper = coords.max(axis=0)

    return tuple(
        slice(
            max(0, int(lower[axis]) - pad),
            min(mask_a.shape[axis], int(upper[axis]) + pad + 1),
        )
        for axis, pad in enumerate(
            max(1, int(math.ceil(max_distance_um / voxel_size_um[axis])) + 4) for axis in range(3)
        )
    )

def _surface(mask: np.ndarray, global_surface_crop: Optional[np.ndarray]) -> np.ndarray:
    if global_surface_crop is not None:
        return mask & global_surface_crop
    return mask & ~binary_erosion(mask, structure=_CONN_26)


def _direct_contact(surface: np.ndarray, mask_other: np.ndarray) -> np.ndarray:
    if not surface.any():
        return np.zeros_like(surface, dtype=bool)
    return surface & binary_dilation(mask_other, structure=_CONN_26)

def _facing_patch(
    surface: np.ndarray,
    dist_to_other_surface: np.ndarray,
    separation_um: float,
    surface_distance_um: float,
) -> np.ndarray:
    if not surface.any() or not math.isfinite(separation_um):
        return np.zeros_like(surface, dtype=bool)
    return surface & (dist_to_other_surface <= separation_um + surface_distance_um)

def _pair_distance_fields(
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    d_a = distance_transform_edt(~surface_a, sampling=voxel_size_um).astype(np.float32)
    d_b = distance_transform_edt(~surface_b, sampling=voxel_size_um).astype(np.float32)
    return d_a, d_b

def _pair_interfaces(
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    d_a: np.ndarray,
    d_b: np.ndarray,
    surface_distance_um: float,
):
    empty = np.zeros_like(surface_a, dtype=bool)
    if not surface_a.any() or not surface_b.any():
        return empty, empty.copy(), empty.copy(), empty.copy(), float('inf'), False

    separation_um = float(d_a[surface_b].min())
    direct_a = _direct_contact(surface_a, mask_b)
    direct_b = _direct_contact(surface_b, mask_a)
    is_direct = bool(direct_a.any() and direct_b.any())
    facing_a = _facing_patch(surface_a, d_b, separation_um, surface_distance_um)
    facing_b = _facing_patch(surface_b, d_a, separation_um, surface_distance_um)
    return direct_a, direct_b, facing_a, facing_b, separation_um, is_direct


def _bridged_corridor(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    mask_crop: np.ndarray,
    d_a: np.ndarray,
    d_b: np.ndarray,
    facing_a: np.ndarray,
    facing_b: np.ndarray,
    max_distance_um: float,
    exclude_other_cells: bool,
    require_bridge: bool,
):
    reachable = (d_a + d_b) <= max_distance_um
    other_cells = (mask_crop != 0) & ~mask_a & ~mask_b
    blocked = bool(exclude_other_cells and (other_cells & reachable).any())

    candidate = ~(mask_a | mask_b) & reachable
    if exclude_other_cells:
        candidate &= ~other_cells

    if not require_bridge:
        return candidate, -1, blocked

    if not candidate.any():
        return candidate, 0, blocked

    labelled, n_components = label(candidate, structure=_CONN_26)
    reach_a = set(np.unique(labelled[binary_dilation(facing_a, structure=_CONN_26)]).tolist())
    reach_b = set(np.unique(labelled[binary_dilation(facing_b, structure=_CONN_26)]).tolist())
    bridges = sorted((reach_a & reach_b) - {0})

    if not bridges:
        return np.zeros_like(candidate), int(n_components), blocked
    if len(bridges) == n_components:
        return candidate, int(n_components), blocked
    return np.isin(labelled, bridges), int(n_components), blocked


def _prune_to_anchored_components(pair_mask: np.ndarray, anchor: np.ndarray):
    if not pair_mask.any():
        return pair_mask, 0

    anchor = anchor & pair_mask
    if not anchor.any():
        return np.zeros_like(pair_mask), 0

    labelled, n_components = label(pair_mask, structure=_CONN_26)
    keep = sorted(set(np.unique(labelled[anchor]).tolist()) - {0})
    if not keep:
        return np.zeros_like(pair_mask), 0
    if len(keep) == n_components:
        return pair_mask, n_components
    return np.isin(labelled, keep), len(keep)


def _cell_territory(
    cell_mask: np.ndarray,
    surface: np.ndarray,
    contact: np.ndarray,
    corridor: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    rho_threshold: float,
    contact_rim_um: float,
    max_inward_um: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = np.zeros_like(cell_mask, dtype=bool)
    rho = np.full(cell_mask.shape, np.nan, dtype=np.float32)

    seed = contact | corridor
    if not cell_mask.any() or not seed.any():
        return empty, rho, np.full(cell_mask.shape, np.inf, dtype=np.float32)

    d_contact = distance_transform_edt(~seed, sampling=voxel_size_um).astype(np.float32)
    non_contact = surface & ~contact & (d_contact > contact_rim_um)

    inside = cell_mask
    if non_contact.any():
        d_other = distance_transform_edt(~non_contact, sampling=voxel_size_um).astype(np.float32)
        total = d_contact[inside] + d_other[inside]
        rho[inside] = np.divide(
            d_contact[inside],
            total,
            out=np.zeros(total.shape, dtype=np.float32),
            where=total > 0,
        )
    else:
        rho[inside] = 0.0

    territory = np.zeros_like(cell_mask, dtype=bool)
    territory[inside] = rho[inside] <= rho_threshold
    if max_inward_um is not None:
        territory[inside] &= d_contact[inside] <= max_inward_um

    return territory, rho, d_contact

# Sparse per-pair encoding

COMPONENT_KEYS = ('territory_a', 'corridor', 'territory_b', 'interface_a', 'interface_b')
INTERFACE_KIND = {'direct': 1, 'near': 2}

def _tight_bbox_of(mask: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    axis_any = [mask.any(axis=tuple(a for a in range(3) if a != axis)) for axis in range(3)]
    if not axis_any[0].any():
        return None
    bounds = []
    for hits in axis_any:
        idx = np.flatnonzero(hits)
        bounds.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(bounds)

def _delta_encode(indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return indices
    out = indices.copy()
    out[1:] = np.diff(indices)
    return out

def encode_component_indices(
    masks: Dict[str, np.ndarray],
    tight: Tuple[slice, slice, slice],
) -> Dict[str, np.ndarray]:
    shape = tuple(int(tight[axis].stop - tight[axis].start) for axis in range(3))
    extent = int(shape[0]) * int(shape[1]) * int(shape[2])
    if extent >= 2 ** 32:
        raise ValueError(
            f"Pair crop {shape} has {extent} voxels, beyond the uint32 index range used by "
            f"the pair archive."
        )
    encoded = {}
    for name, mask in masks.items():
        flat = np.flatnonzero(np.ascontiguousarray(mask[tight]).ravel())
        encoded[name] = _delta_encode(flat.astype(np.uint32, copy=False))
    return encoded

def decode_component_mask(
    indices: np.ndarray,
    shape: Tuple[int, int, int],
) -> np.ndarray:
    mask = np.zeros(int(shape[0]) * int(shape[1]) * int(shape[2]), dtype=bool)
    if indices.size:
        mask[np.cumsum(indices, dtype=np.uint64)] = True
    return mask.reshape(tuple(int(v) for v in shape))

def _mask_geometry(
    mask: np.ndarray,
    origin_zyx: Tuple[int, int, int],
    voxel_size_um: Tuple[float, float, float],
):
    coords = np.argwhere(mask)
    if coords.size == 0:
        nan3 = (float('nan'),) * 3
        return nan3, nan3, float('nan'), float('nan')

    scale = np.asarray(voxel_size_um, dtype=np.float64)
    global_voxel = coords.astype(np.float64) + np.asarray(origin_zyx, dtype=np.float64)
    centroid_voxel = global_voxel.mean(axis=0)
    radii = np.sqrt((((global_voxel - centroid_voxel) * scale) ** 2).sum(axis=1))
    return (
        tuple(float(v) for v in centroid_voxel),
        tuple(float(v) for v in centroid_voxel * scale),
        float(radii.min()),
        float(radii.max()),
    )

def compute_interscellar_volume_adaptive(
    mask_3d: np.ndarray,
    cell_a_id: int,
    cell_b_id: int,
    voxel_size_um: Tuple[float, float, float] = (0.56, 0.28, 0.28),
    global_surface: Optional[np.ndarray] = None,
    halo_bboxes: Optional[Dict[int, Tuple[slice, slice, slice]]] = None,
    max_distance_um: float = 3.0,
    surface_distance_um: float = 0.5,
    rho_threshold: float = 0.5,
    contact_rim_um: float = 0.0,
    max_inward_um: Optional[float] = None,
    exclude_other_cells: bool = True,
    reject_unbridged: bool = True,
    pair_id: Optional[int] = None,
    return_debug: bool = False,
) -> Dict[str, Any]:

    if cell_a_id == cell_b_id:
        raise ValueError("cell_a_id and cell_b_id must be different for a pair computation")

    if halo_bboxes is None:
        if not isinstance(mask_3d, np.ndarray):
            raise ValueError(
                f"halo_bboxes is required when the label volume is read lazily "
                f"({type(mask_3d).__name__}): finding cells {cell_a_id} and {cell_b_id} "
                f"would scan the whole volume. Pass halo_bboxes=compute_halo_bboxes(...), "
                f"or load the volume with _load_label_volume() first."
            )
        mask_a_full = mask_3d == cell_a_id
        mask_b_full = mask_3d == cell_b_id
        if not mask_a_full.any() or not mask_b_full.any():
            raise ValueError(f"Cells {cell_a_id} and {cell_b_id} are not present in the label volume")
        union_bbox = _fallback_union_bbox(mask_a_full, mask_b_full, voxel_size_um, max_distance_um)
    else:
        union_bbox = _pair_union_bbox(halo_bboxes, cell_a_id, cell_b_id)

    mask_crop = mask_3d[union_bbox]
    mask_a = mask_crop == cell_a_id
    mask_b = mask_crop == cell_b_id

    if not mask_a.any() or not mask_b.any():
        raise ValueError(f"Pair {cell_a_id}-{cell_b_id} is missing a cell mask in the local crop")

    global_surface_crop = None if global_surface is None else global_surface[union_bbox]
    surface_a = _surface(mask_a, global_surface_crop)
    surface_b = _surface(mask_b, global_surface_crop)

    # 1. One pair of surface distance fields, reused by everything below
    if surface_a.any() and surface_b.any():
        d_a, d_b = _pair_distance_fields(surface_a, surface_b, voxel_size_um)
    else:
        inf = np.full(mask_crop.shape, np.inf, dtype=np.float32)
        d_a, d_b = inf, inf

    voxel_volume_um3 = float(np.prod(voxel_size_um))
    params = {
        'voxel_volume_um3': voxel_volume_um3,
        'max_distance_threshold_um': max_distance_um,
        'surface_distance_um': surface_distance_um,
        'rho_threshold': rho_threshold,
        'contact_rim_um': contact_rim_um,
        'max_inward_um': max_inward_um,
    }

    # 2. Both interface entities, and the pair's adjacency class
    direct_a, direct_b, facing_a, facing_b, separation_um, is_direct = _pair_interfaces(
        surface_a, surface_b, mask_a, mask_b, d_a, d_b, surface_distance_um
    )

    # 3. Extracellular corridor, keeping only components that bridge the two facing patches
    corridor, n_candidate_components, blocked = _bridged_corridor(
        mask_a, mask_b, mask_crop, d_a, d_b, facing_a, facing_b,
        max_distance_um, exclude_other_cells, require_bridge=reject_unbridged,
    )

    # 4. Reject unbridged near pairs
    if reject_unbridged and not is_direct and not corridor.any():
        return {
            'cell_a_id': cell_a_id,
            'cell_b_id': cell_b_id,
            'pair_id': pair_id,
            'rejected': 'unbridged_near_pair',
            'interface_kind': INTERFACE_KIND['near'],
            'min_surface_separation_um': separation_um,
            'n_candidate_components': n_candidate_components,
            'blocked_by_other_cell': blocked,
            **params,
        }

    # 5. Adaptive intracellular territory
    interface_region_a = direct_a | facing_a
    interface_region_b = direct_b | facing_b
    territory_a, rho_a, d_contact_a = _cell_territory(
        mask_a, surface_a, interface_region_a, corridor,
        voxel_size_um, rho_threshold, contact_rim_um, max_inward_um,
    )
    territory_b, rho_b, d_contact_b = _cell_territory(
        mask_b, surface_b, interface_region_b, corridor,
        voxel_size_um, rho_threshold, contact_rim_um, max_inward_um,
    )

    # 6. Union, then drop any component with no interface anchor
    interscellar_mask = territory_a | corridor | territory_b
    n_components = -1
    if reject_unbridged:
        anchor = corridor | direct_a | direct_b
        interscellar_mask, n_components = _prune_to_anchored_components(interscellar_mask, anchor)
        territory_a = territory_a & interscellar_mask
        territory_b = territory_b & interscellar_mask
        corridor = corridor & interscellar_mask

    if reject_unbridged and not interscellar_mask.any():
        return {
            'cell_a_id': cell_a_id,
            'cell_b_id': cell_b_id,
            'pair_id': pair_id,
            'rejected': 'empty_after_pruning',
            'interface_kind': INTERFACE_KIND['direct'] if is_direct else INTERFACE_KIND['near'],
            'min_surface_separation_um': separation_um,
            'n_candidate_components': n_candidate_components,
            'blocked_by_other_cell': blocked,
            **params,
        }

    # 7. Archived interface: contact (touching) or facing (near) patch
    if is_direct:
        interface_a, interface_b = direct_a & interscellar_mask, direct_b & interscellar_mask
    else:
        interface_a, interface_b = facing_a & interscellar_mask, facing_b & interscellar_mask

    # 8. Three components are disjoint
    interscellar_voxels = int(interscellar_mask.sum())
    interscellar_volume_um3 = interscellar_voxels * voxel_volume_um3

    tight = _tight_bbox_of(interscellar_mask)
    origin_zyx = tuple(int(union_bbox[axis].start + tight[axis].start) for axis in range(3))
    shape_zyx = tuple(int(tight[axis].stop - tight[axis].start) for axis in range(3))
    centroid_voxel, centroid_um, r_min_um, r_max_um = _mask_geometry(
        interscellar_mask[tight], origin_zyx, voxel_size_um
    )

    result: Dict[str, Any] = {
        'cell_a_id': cell_a_id,
        'cell_b_id': cell_b_id,
        'pair_id': pair_id,
        'union_bbox': union_bbox,
        'interscellar_mask': interscellar_mask,
        'interscellar_voxels': interscellar_voxels,
        'interscellar_volume_um3': interscellar_volume_um3,
        'total_interscellar_volume_voxels': interscellar_voxels,
        'total_interscellar_volume_um3': interscellar_volume_um3,
        # Component breakdown (disjoint; sum to total)
        'corridor_voxels': int(corridor.sum()),
        'territory_a_voxels': int(territory_a.sum()),
        'territory_b_voxels': int(territory_b.sum()),
        # Interface entities
        'interface_kind': INTERFACE_KIND['direct'] if is_direct else INTERFACE_KIND['near'],
        'is_direct_contact': is_direct,
        'n_interface_a': int(interface_a.sum()),
        'n_interface_b': int(interface_b.sum()),
        # Full facing band
        'seed_a_voxels': int(interface_region_a.sum()),
        'seed_b_voxels': int(interface_region_b.sum()),
        # Connectivity and bridge diagnostics
        'min_surface_separation_um': separation_um,
        'n_components': n_components,
        'n_candidate_components': n_candidate_components,
        'blocked_by_other_cell': blocked,
        # Geometry precomputed for scoring
        'origin_zyx': origin_zyx,
        'shape_zyx': shape_zyx,
        'centroid_z_voxel': centroid_voxel[0],
        'centroid_y_voxel': centroid_voxel[1],
        'centroid_x_voxel': centroid_voxel[2],
        'centroid_z_um': centroid_um[0],
        'centroid_y_um': centroid_um[1],
        'centroid_x_um': centroid_um[2],
        'r_min_um': r_min_um,
        'r_max_um': r_max_um,
        **params,
    }

    result['component_indices'] = encode_component_indices(
        {
            'territory_a': territory_a,
            'corridor': corridor,
            'territory_b': territory_b,
            'interface_a': interface_a,
            'interface_b': interface_b,
        },
        tight,
    )

    if pair_id is not None:
        result['labeled_pair_id'] = int(pair_id)

    if return_debug:
        result.update({
            'corridor_mask': corridor,
            'adaptive_a_mask': territory_a,
            'adaptive_b_mask': territory_b,
            'direct_contact_a': direct_a,
            'direct_contact_b': direct_b,
            'facing_patch_a': facing_a,
            'facing_patch_b': facing_b,
            'contact_surface_a': interface_a,
            'contact_surface_b': interface_b,
            'rho_a': rho_a,
            'rho_b': rho_b,
            'd_contact_a': d_contact_a,
            'd_contact_b': d_contact_b,
        })

    return result

compute_interscellar_volume_adaptive_for_pair = compute_interscellar_volume_adaptive

# Batch driver

def truncated_cell_ids(mask_3d: np.ndarray) -> set:
    faces = (
        mask_3d[0], mask_3d[-1],
        mask_3d[:, 0], mask_3d[:, -1],
        mask_3d[:, :, 0], mask_3d[:, :, -1],
    )
    ids = set()
    for face in faces:
        ids.update(np.unique(face).tolist())
    ids.discard(0)
    return ids

def compute_interscellar_volumes_adaptive_for_pairs(
    mask_3d: np.ndarray,
    pairs,
    voxel_size_um: Tuple[float, float, float] = (0.56, 0.28, 0.28),
    global_surface: Optional[np.ndarray] = None,
    halo_bboxes: Optional[Dict[int, Tuple[slice, slice, slice]]] = None,
    exclude_truncated: bool = False,
    keep_masks: bool = True,
    verbose: bool = True,
    **kwargs: Any,
):
    present = set(np.unique(mask_3d).tolist())
    present.discard(0)

    dropped: Dict[str, list] = {'cell_absent': [], 'truncated': [], 'failed': [], 'rejected': []}
    if exclude_truncated:
        edge_ids = truncated_cell_ids(mask_3d)
        if verbose:
            print(f"Cells touching a volume face: {len(edge_ids)}")
    else:
        edge_ids = set()

    records = []
    for cell_a_id, cell_b_id, pair_id in pairs:
        cell_a_id, cell_b_id = int(cell_a_id), int(cell_b_id)

        if cell_a_id not in present or cell_b_id not in present:
            dropped['cell_absent'].append((cell_a_id, cell_b_id, pair_id))
            continue
        if cell_a_id in edge_ids or cell_b_id in edge_ids:
            dropped['truncated'].append((cell_a_id, cell_b_id, pair_id))
            continue

        try:
            result = compute_interscellar_volume_adaptive(
                mask_3d=mask_3d,
                cell_a_id=cell_a_id,
                cell_b_id=cell_b_id,
                voxel_size_um=voxel_size_um,
                global_surface=global_surface,
                halo_bboxes=halo_bboxes,
                pair_id=None if pair_id is None else int(pair_id),
                **kwargs,
            )
        except (ValueError, KeyError) as exc:
            dropped['failed'].append((cell_a_id, cell_b_id, pair_id, str(exc)))
            continue

        if 'rejected' in result:
            dropped['rejected'].append(result)
            continue

        if not keep_masks:
            result = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        records.append(result)

    if verbose:
        print(
            f"Computed {len(records)} pairs | skipped: "
            f"{len(dropped['cell_absent'])} not in volume, "
            f"{len(dropped['truncated'])} truncated, "
            f"{len(dropped['rejected'])} unbridged, "
            f"{len(dropped['failed'])} failed"
        )
    return records, dropped

def _on_disk_zarr_format(path: str) -> Optional[int]:
    if os.path.exists(os.path.join(path, "zarr.json")):
        return 3
    if os.path.exists(os.path.join(path, ".zgroup")) or os.path.exists(
        os.path.join(path, ".zarray")
    ):
        return 2
    return None

def _open_zarr_store(path: str, what: str = "label volume"):
    import zarr

    try:
        return zarr.open(path, mode="r")
    except Exception as exc:
        if _on_disk_zarr_format(path) == 3 and int(str(zarr.__version__).split(".")[0]) < 3:
            raise SystemExit(
                f"The {what} at {path} is a Zarr v3 store, but this environment has "
                f"zarr {zarr.__version__}, which reads only v2. Run from an environment "
                f"with zarr>=3 (which requires Python >=3.11); the store was most likely "
                f"written by one."
            ) from exc
        raise

class _Lazy3DView:

    def __init__(self, array):
        self._array = array
        self._lead = (0,) * (int(array.ndim) - 3)
        self.shape = tuple(int(v) for v in array.shape[-3:])
        self.ndim = 3
        self.dtype = np.dtype(array.dtype).newbyteorder("=")
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        block = np.asarray(self._array[self._lead + key])
        if block.dtype.byteorder == ">":
            block = block.astype(block.dtype.newbyteorder("="))
        return block


def _open_label_volume_lazy(path: str):
    if path.endswith(".npy"):
        array = np.load(path, mmap_mode="r")
        return array if array.ndim == 3 else None

    group = _open_zarr_store(path)
    if hasattr(group, "shape"):
        array = group
    elif "labels" in group:
        array = group["labels"]
    elif "0" in group and not hasattr(group["0"], "shape"):
        array = group["0"]["0"]
    elif "0" in group:
        array = group["0"]
    else:
        try:
            names = list(group.array_keys()) if hasattr(group, "array_keys") else list(group.keys())
        except Exception:
            return None
        array = next((group[k] for k in names
                      if hasattr(group[k], "shape") and len(group[k].shape) >= 3), None)
    ndim = int(getattr(array, "ndim", 0))
    if array is None or ndim < 3 or ndim > 5:
        return None
    return array if ndim == 3 else _Lazy3DView(array)


def _load_label_volume(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)

    group = _open_zarr_store(path)
    if "labels" in group:
        array = group["labels"][0, 0]
    elif "0" in group and hasattr(group["0"], "keys") and "0" in group["0"]:
        array = group["0"]["0"][0, 0]
    else:
        array = None
        for key in group.keys():
            candidate = group[key]
            if hasattr(candidate, "shape") and len(candidate.shape) >= 3:
                array = np.asarray(candidate)
                break
        if array is None:
            raise ValueError(f"Could not find a 3D label array in {path}")

    array = np.asarray(array)
    while array.ndim > 3:
        array = array[0]
    if array.dtype.byteorder == ">":
        native = array.dtype.newbyteorder("=")
        if array.flags.writeable and array.flags.owndata:
            array = array.byteswap(inplace=True).view(native)
        else:
            array = array.astype(native)
    return array


def _load_pairs(path: str):
    import pandas as pd

    if path.endswith(".db"):
        import sqlite3
        conn = sqlite3.connect(path)
        try:
            df = pd.read_sql_query(
                "SELECT cell_id_a AS cell_a_id, cell_id_b AS cell_b_id, pair_id "
                "FROM neighbors ORDER BY pair_id",
                conn,
            )
        finally:
            conn.close()
    else:
        df = pd.read_csv(path)
        df = df.rename(columns={'cell_id_a': 'cell_a_id', 'cell_id_b': 'cell_b_id'})
        missing = {'cell_a_id', 'cell_b_id'} - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing {sorted(missing)}. Found columns: {sorted(df.columns)}"
            )

    if 'pair_id' not in df.columns or df['pair_id'].isna().all():
        df = df.reset_index(drop=True)
        df['pair_id'] = np.arange(1, len(df) + 1)

    df = df.dropna(subset=['cell_a_id', 'cell_b_id', 'pair_id'])
    return [
        (int(a), int(b), int(p))
        for a, b, p in zip(df['cell_a_id'], df['cell_b_id'], df['pair_id'])
    ]

# Parallel execution

_WORKER: Dict[str, Any] = {}

def _init_worker(mask_path: str, kwargs: Dict[str, Any], keep_mask: bool, lazy: bool = True) -> None:
    volume = _open_label_volume_lazy(mask_path) if lazy else None
    _WORKER['mask'] = volume if volume is not None else _load_label_volume(mask_path)
    _WORKER['kwargs'] = kwargs
    _WORKER['keep_mask'] = keep_mask


def _worker_batch(batch):
    mask, kwargs, keep_mask = _WORKER['mask'], _WORKER['kwargs'], _WORKER['keep_mask']
    out = []
    for cell_a_id, cell_b_id, pair_id in batch:
        try:
            result = compute_interscellar_volume_adaptive(
                mask_3d=mask, cell_a_id=cell_a_id, cell_b_id=cell_b_id,
                pair_id=pair_id, **kwargs,
            )
        except (ValueError, KeyError) as exc:
            out.append({'cell_a_id': cell_a_id, 'cell_b_id': cell_b_id,
                        'pair_id': pair_id, 'error': str(exc)})
            continue
        slim = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        if not keep_mask:
            slim.pop('component_indices', None)
        out.append(slim)
    return out


def _run_pairs_parallel(mask_path, pairs, n_jobs, keep_mask, kwargs, verbose=True, lazy=True):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_batches = max(n_jobs * 4, 1)
    batches = [pairs[i::n_batches] for i in range(n_batches)]
    batches = [b for b in batches if b]

    records, done = [], 0
    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_init_worker,
        initargs=(mask_path, kwargs, keep_mask, lazy),
    ) as pool:
        futures = [pool.submit(_worker_batch, b) for b in batches]
        for future in as_completed(futures):
            batch_records = future.result()
            records.extend(batch_records)
            done += 1
            if verbose:
                print(f"  batch {done}/{len(futures)} done -- {len(records)} pairs computed", flush=True)

    return records

def compute_halo_bboxes(
    mask_3d: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    max_distance_um: float,
    cell_ids=None,
) -> Dict[int, Tuple[slice, slice, slice]]:
    from scipy.ndimage import find_objects

    try:
        objects = find_objects(mask_3d)
    except (TypeError, RuntimeError):
        objects = find_objects(mask_3d.astype(np.int32))

    pads = [max(1, int(math.ceil(max_distance_um / voxel_size_um[axis])) + 4) for axis in range(3)]
    wanted = None if cell_ids is None else set(int(c) for c in cell_ids)

    bboxes: Dict[int, Tuple[slice, slice, slice]] = {}
    for index, box in enumerate(objects):
        if box is None:
            continue
        label = index + 1
        if wanted is not None and label not in wanted:
            continue
        bboxes[label] = tuple(
            slice(
                max(0, box[axis].start - pads[axis]),
                min(mask_3d.shape[axis], box[axis].stop + pads[axis]),
            )
            for axis in range(3)
        )
    return bboxes

CSV_COLUMNS = [
    'pair_id', 'cell_a_id', 'cell_b_id',
    'total_interscellar_volume_um3', 'total_interscellar_volume_voxels',
    'territory_a_voxels', 'corridor_voxels', 'territory_b_voxels',
    'interface_kind', 'is_direct_contact', 'n_interface_a', 'n_interface_b',
    'min_surface_separation_um', 'n_components', 'n_candidate_components',
    'blocked_by_other_cell',
    'seed_a_voxels', 'seed_b_voxels',
    'centroid_z_um', 'centroid_y_um', 'centroid_x_um',
    'centroid_z_voxel', 'centroid_y_voxel', 'centroid_x_voxel',
    'r_min_um', 'r_max_um',
    'shared_voxels', 'exclusive_voxels', 'voxel_volume_um3',
    'max_distance_threshold_um', 'surface_distance_um', 'rho_threshold',
    'contact_rim_um', 'max_inward_um',
]

REJECTED_COLUMNS = [
    'pair_id', 'cell_a_id', 'cell_b_id', 'rejected', 'interface_kind',
    'min_surface_separation_um', 'n_candidate_components', 'blocked_by_other_cell',
    'max_distance_threshold_um', 'surface_distance_um',
]

def _create_array(store, name, shape, dtype, chunks, fill_value=0):
    if hasattr(store, "create_array"):          # zarr 3
        try:
            return store.create_array(name, shape=shape, dtype=dtype, chunks=chunks,
                                      fill_value=fill_value)
        except TypeError:
            return store.create_array(name, shape=shape, dtype=dtype, chunks=chunks)
    return store.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks,   # zarr 2
                                fill_value=fill_value)

def _open_output_zarr(path, shape, appending, voxel_size_um, geometry, preview=True):
    import zarr

    store = zarr.open(path, mode="a" if appending else "w")
    if preview and "interscellar_meshes" not in store:
        chunks = (min(32, shape[0]), min(256, shape[1]), min(256, shape[2]))
        _create_array(store, "interscellar_meshes", shape, "uint32", chunks)
        _create_array(store, "overlap_count", shape, "uint16", chunks)
    store.attrs["description"] = "Adaptive interscellar volumes labeled by pair ID"
    store.attrs["contains"] = "interscellar volumes only; cells are not written"
    store.attrs["store_format_version"] = 2
    store.attrs["authoritative"] = _ARCHIVE_GROUP
    store.attrs["lossless_note"] = (
        f"{_ARCHIVE_GROUP} holds every pair's complete footprint, shared voxels included. "
        f"interscellar_meshes and overlap_count are previews and cannot represent overlap."
    )
    store.attrs["interscellar_meshes_role"] = "preview: highest pair_id per voxel"
    store.attrs["overlap_count_role"] = "preview: number of pairs claiming each voxel"
    store.attrs["label_collision_policy"] = (
        "highest pair_id wins (np.maximum), matching compute_interscellar_volumes_3d"
    )
    store.attrs["overlap_count_meaning"] = (
        "number of pairs claiming each voxel; >1 means the pair ID shown is one of several"
    )
    store.attrs["voxel_size_um"] = list(voxel_size_um)
    store.attrs["coordinate_system"] = "same_as_input_segmentation"
    store.attrs["axes"] = ["z", "y", "x"]
    if appending:
        stored = {k: store.attrs.get(k) for k in geometry if store.attrs.get(k) is not None}
        wanted = {k: v for k, v in geometry.items() if v is not None}
        drift = {k: (stored.get(k), wanted[k]) for k in wanted if k in stored and stored[k] != wanted[k]}
        stored_voxel = store.attrs.get("voxel_size_um")
        if stored_voxel is not None and list(stored_voxel) != list(voxel_size_um):
            drift['voxel_size_um'] = (stored_voxel, list(voxel_size_um))
        if drift:
            detail = "; ".join(f"{k}: existing {old} vs requested {new}" for k, (old, new) in sorted(drift.items()))
            raise SystemExit(
                f"Refusing to resume into {path}: it was written with different parameters "
                f"({detail}). Use a new --out/--out-zarr, or drop --resume to start over."
            )

    store.attrs.update({k: v for k, v in geometry.items() if v is not None})

    archive = PairVolumeArchive.open(store, appending)
    ds_labels = store["interscellar_meshes"] if "interscellar_meshes" in store else None
    ds_overlap = store["overlap_count"] if "overlap_count" in store else None

    written = archive.written_pair_ids() if appending else set()
    return store, ds_labels, ds_overlap, written, archive


def read_pair(store_or_path, pair_id: int) -> Dict[str, Any]:
    import zarr

    store = zarr.open(store_or_path, mode="r") if isinstance(store_or_path, str) else store_or_path
    group = store[_ARCHIVE_GROUP]
    n_rows = int(group.attrs.get("committed_rows", group['pair_id'].shape[0]))

    ids = np.asarray(group['pair_id'][:n_rows])
    if ids.size and bool(np.all(np.diff(ids.astype(np.int64)) > 0)):
        row = int(np.searchsorted(ids, pair_id))
        if row >= ids.size or int(ids[row]) != int(pair_id):
            row = -1
    else:
        hits = np.flatnonzero(ids == pair_id)
        row = int(hits[0]) if hits.size else -1
    if row < 0:
        raise KeyError(
            f"pair_id {pair_id} is not in this archive. It was either never computed, or "
            f"rejected as unbridged -- check the rejected-pairs CSV."
        )

    shape = tuple(int(v) for v in np.asarray(group['shape_zyx'][row]))
    out: Dict[str, Any] = {
        'pair_id': int(pair_id),
        'cell_a_id': int(group['cell_a_id'][row]),
        'cell_b_id': int(group['cell_b_id'][row]),
        'interface_kind': int(group['interface_kind'][row]),
        'origin_zyx': tuple(int(v) for v in np.asarray(group['origin_zyx'][row])),
        'shape_zyx': shape,
        'n_voxels': int(group['n_voxels'][row]),
        'n_components': int(group['n_components'][row]),
        'shared_voxels': int(group['shared_voxels'][row]),
        'centroid_zyx_voxel': tuple(float(v) for v in np.asarray(group['centroid_zyx_voxel'][row])),
        'centroid_zyx_um': tuple(float(v) for v in np.asarray(group['centroid_zyx_um'][row])),
        'r_min_um': float(group['r_min_um'][row]),
        'r_max_um': float(group['r_max_um'][row]),
        'min_surface_separation_um': float(group['min_surface_separation_um'][row]),
        'volume_um3': float(group['volume_um3'][row]),
    }
    out['interface_kind_name'] = 'direct' if out['interface_kind'] == 1 else 'near'

    for name in COMPONENT_KEYS:
        lo, hi = (int(v) for v in np.asarray(group[f"{name}_indptr"][row:row + 2]))
        indices = np.asarray(group[name][lo:hi])
        out[name] = decode_component_mask(indices, shape)

    # The three volume components are disjoint, so OR-ing them is the exact footprint.
    out['mask'] = out['territory_a'] | out['corridor'] | out['territory_b']
    return out

def _write_pair_into(ds_labels, ds_overlap, record):
    origin, shape = record['origin_zyx'], record['shape_zyx']
    bbox = tuple(slice(int(origin[axis]), int(origin[axis] + shape[axis])) for axis in range(3))
    flat = _pair_flat_indices(record)
    pair_id = np.uint32(int(record['pair_id']))

    region = np.asarray(ds_labels[bbox]).ravel()
    region[flat] = np.maximum(region[flat], pair_id)
    ds_labels[bbox] = region.reshape(tuple(int(v) for v in shape))

    counts = np.asarray(ds_overlap[bbox]).ravel()
    # The indices are unique within a pair, so a buffered fancy-index increment is exact.
    counts[flat] += 1
    ds_overlap[bbox] = counts.reshape(tuple(int(v) for v in shape))

def _pair_flat_indices(record) -> np.ndarray:
    encoded = record['component_indices']
    parts = [
        np.cumsum(encoded[name], dtype=np.uint64)
        for name in ('territory_a', 'corridor', 'territory_b')
        if encoded[name].size
    ]
    if not parts:
        return np.zeros(0, dtype=np.uint64)
    flat = np.concatenate(parts)
    flat.sort()
    return flat

# Pair archive: lossless

_ARCHIVE_GROUP = "pair_volumes"

_ARCHIVE_ROW_FIELDS = (
    ('pair_id', 'uint32', 1),
    ('cell_a_id', 'uint32', 1),
    ('cell_b_id', 'uint32', 1),
    ('interface_kind', 'uint8', 1),
    ('origin_zyx', 'int32', 3),
    ('shape_zyx', 'int32', 3),
    ('n_voxels', 'uint32', 1),
    ('n_territory_a', 'uint32', 1),
    ('n_corridor', 'uint32', 1),
    ('n_territory_b', 'uint32', 1),
    ('n_interface_a', 'uint32', 1),
    ('n_interface_b', 'uint32', 1),
    ('n_components', 'int32', 1),
    ('centroid_zyx_voxel', 'float64', 3),
    ('centroid_zyx_um', 'float64', 3),
    ('r_min_um', 'float32', 1),
    ('r_max_um', 'float32', 1),
    ('min_surface_separation_um', 'float32', 1),
    ('volume_um3', 'float64', 1),
    ('shared_voxels', 'uint32', 1),
)

_ROW_FROM_RECORD = {
    'pair_id': lambda r: int(r['pair_id']),
    'cell_a_id': lambda r: int(r['cell_a_id']),
    'cell_b_id': lambda r: int(r['cell_b_id']),
    'interface_kind': lambda r: int(r['interface_kind']),
    'origin_zyx': lambda r: [int(v) for v in r['origin_zyx']],
    'shape_zyx': lambda r: [int(v) for v in r['shape_zyx']],
    'n_voxels': lambda r: int(r['interscellar_voxels']),
    'n_territory_a': lambda r: int(r['territory_a_voxels']),
    'n_corridor': lambda r: int(r['corridor_voxels']),
    'n_territory_b': lambda r: int(r['territory_b_voxels']),
    'n_interface_a': lambda r: int(r['n_interface_a']),
    'n_interface_b': lambda r: int(r['n_interface_b']),
    'n_components': lambda r: int(r['n_components']),
    'centroid_zyx_voxel': lambda r: [r['centroid_z_voxel'], r['centroid_y_voxel'],
                                     r['centroid_x_voxel']],
    'centroid_zyx_um': lambda r: [r['centroid_z_um'], r['centroid_y_um'], r['centroid_x_um']],
    'r_min_um': lambda r: r['r_min_um'],
    'r_max_um': lambda r: r['r_max_um'],
    'min_surface_separation_um': lambda r: r['min_surface_separation_um'],
    'volume_um3': lambda r: float(r['total_interscellar_volume_um3']),
    'shared_voxels': lambda r: 0,
}

def _resize_array(store, name, length):
    array = store[name]
    shape = (int(length),) + tuple(array.shape[1:])
    resized = array.resize(shape)
    return store[name] if resized is None else resized

class PairVolumeArchive:

    def __init__(self, store, group):
        self.store = store
        self.group = group
        self.n_rows = int(group.attrs.get("committed_rows", 0))
        self.lengths = {
            name: int(group.attrs.get(f"committed_{name}", 0)) for name in COMPONENT_KEYS
        }

    @classmethod
    def open(cls, store, appending: bool):
        if _ARCHIVE_GROUP in store:
            group = store[_ARCHIVE_GROUP]
        else:
            group = store.create_group(_ARCHIVE_GROUP)

        for name, dtype, width in _ARCHIVE_ROW_FIELDS:
            if name in group:
                continue
            shape = (0,) if width == 1 else (0, width)
            chunks = (4096,) if width == 1 else (4096, width)
            _create_array(group, name, shape, dtype, chunks)
        for name in COMPONENT_KEYS:
            if f"{name}_indptr" not in group:
                # indptr[0] = 0 always, so a committed row count of n means n+1 entries.
                _create_array(group, f"{name}_indptr", (1,), "uint64", (4096,))
            if name not in group:
                # 1 MiB of raw uint32 per chunk: big enough to stream, small enough that
                # pulling one pair's row decompresses only a chunk or two.
                _create_array(group, name, (0,), "uint32", (262144,))

        archive = cls(store, group)
        if appending:
            archive._truncate_to_commit()
        else:
            archive.n_rows = 0
            archive.lengths = {name: 0 for name in COMPONENT_KEYS}
            archive._commit()
        archive._write_schema_attrs()
        return archive

    def _truncate_to_commit(self):
        for name, _dtype, _width in _ARCHIVE_ROW_FIELDS:
            if self.group[name].shape[0] != self.n_rows:
                _resize_array(self.group, name, self.n_rows)
        for name in COMPONENT_KEYS:
            if self.group[f"{name}_indptr"].shape[0] != self.n_rows + 1:
                _resize_array(self.group, f"{name}_indptr", self.n_rows + 1)
            if self.group[name].shape[0] != self.lengths[name]:
                _resize_array(self.group, name, self.lengths[name])

    def _write_schema_attrs(self):
        self.group.attrs["role"] = "authoritative lossless per-pair footprints"
        self.group.attrs["voxel_encoding"] = "delta_local_flat_c_order_uint32"
        self.group.attrs["index_frame"] = "C-order flat within origin_zyx .. origin_zyx+shape_zyx"
        self.group.attrs["components"] = list(COMPONENT_KEYS)
        self.group.attrs["disjoint_components"] = ["territory_a", "corridor", "territory_b"]
        self.group.attrs["interface_kind_enum"] = {"direct": 1, "near": 2}
        self.group.attrs["interface_note"] = (
            "interface_a/b holds direct contact surfaces when interface_kind==1 and facing "
            "patches when interface_kind==2; it is a subset of the territories, not a "
            "fourth disjoint component"
        )
        self.group.attrs["axes"] = ["z", "y", "x"]

    def _commit(self):
        self.group.attrs["committed_rows"] = int(self.n_rows)
        for name in COMPONENT_KEYS:
            self.group.attrs[f"committed_{name}"] = int(self.lengths[name])

    def written_pair_ids(self) -> set:
        if self.n_rows == 0:
            return set()
        return set(int(v) for v in np.asarray(self.group['pair_id'][: self.n_rows]).tolist())

    def append(self, records) -> int:
        rows = [r for r in records if r.get('component_indices') is not None]
        if not rows:
            return 0
        # Keep pair_id ascending on disk so a reader can binary-search it.
        rows.sort(key=lambda r: int(r['pair_id']))

        base = self.n_rows
        new_rows = len(rows)
        for name, _dtype, width in _ARCHIVE_ROW_FIELDS:
            array = _resize_array(self.group, name, base + new_rows)
            build = _ROW_FROM_RECORD[name]
            values = [build(r) for r in rows]
            array[base:base + new_rows] = np.asarray(values).reshape(
                (new_rows,) if width == 1 else (new_rows, width)
            )

        for name in COMPONENT_KEYS:
            chunks = [r['component_indices'][name] for r in rows]
            counts = np.asarray([c.size for c in chunks], dtype=np.uint64)
            start = self.lengths[name]
            total = int(counts.sum())
            if total:
                data = _resize_array(self.group, name, start + total)
                data[start:start + total] = np.concatenate(chunks)
            indptr = _resize_array(self.group, f"{name}_indptr", base + new_rows + 1)
            indptr[base + 1:base + new_rows + 1] = start + np.cumsum(counts)
            self.lengths[name] = start + total

        self.n_rows = base + new_rows
        self._commit()
        return new_rows

    def fill_shared_voxels(self, ds_overlap, verbose: bool = True) -> None:
        if self.n_rows == 0:
            return
        origins = np.asarray(self.group['origin_zyx'][: self.n_rows])
        shapes = np.asarray(self.group['shape_zyx'][: self.n_rows])
        shared = np.zeros(self.n_rows, dtype=np.uint32)
        for row in range(self.n_rows):
            flat = self._footprint_indices(row)
            if flat.size == 0:
                continue
            origin, shape = origins[row], shapes[row]
            bbox = tuple(slice(int(origin[a]), int(origin[a] + shape[a])) for a in range(3))
            counts = np.asarray(ds_overlap[bbox]).ravel()
            shared[row] = int((counts[flat] > 1).sum())
            if verbose and self.n_rows > 200 and row % max(1, self.n_rows // 20) == 0:
                print(f"  shared-voxel pass: {row}/{self.n_rows} rows", end="\r", flush=True)
        self.group['shared_voxels'][: self.n_rows] = shared
        if verbose:
            print(f"  shared-voxel pass: {self.n_rows}/{self.n_rows} rows")

    def _footprint_indices(self, row: int) -> np.ndarray:
        parts = []
        for name in ('territory_a', 'corridor', 'territory_b'):
            lo, hi = np.asarray(self.group[f"{name}_indptr"][row:row + 2]).tolist()
            if hi > lo:
                parts.append(np.cumsum(
                    np.asarray(self.group[name][int(lo):int(hi)]), dtype=np.uint64
                ))
        if not parts:
            return np.zeros(0, dtype=np.uint64)
        flat = np.concatenate(parts)
        flat.sort()
        return flat

def patch_csv_shared_voxels(csv_path: str, archive: "PairVolumeArchive") -> int:
    import csv as _csv
    import tempfile

    if not csv_path or not os.path.exists(csv_path) or archive.n_rows == 0:
        return 0

    ids = np.asarray(archive.group['pair_id'][: archive.n_rows])
    shared = np.asarray(archive.group['shared_voxels'][: archive.n_rows])
    lookup = {int(p): int(s) for p, s in zip(ids.tolist(), shared.tolist())}

    with open(csv_path, newline="") as handle:
        rows = list(_csv.DictReader(handle))

    patched = 0
    for row in rows:
        try:
            pair_id = int(row['pair_id'])
        except (KeyError, TypeError, ValueError):
            continue
        if pair_id not in lookup:
            continue
        total = int(float(row.get('total_interscellar_volume_voxels') or 0))
        row['shared_voxels'] = lookup[pair_id]
        row['exclusive_voxels'] = total - lookup[pair_id]
        patched += 1

    directory = os.path.dirname(os.path.abspath(csv_path)) or "."
    handle, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".csv")
    os.close(handle)
    try:
        with open(tmp, "w", newline="") as out:
            writer = _csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, csv_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return patched

def _report_overlap(ds_labels, ds_overlap, path, slab=16):
    claimed = shared = 0
    peak = 0
    for z in range(0, ds_overlap.shape[0], slab):
        counts = np.asarray(ds_overlap[z:z + slab])
        claimed += int((counts > 0).sum())
        shared += int((counts > 1).sum())
        peak = max(peak, int(counts.max()) if counts.size else 0)
    pct = 100.0 * shared / claimed if claimed else 0.0
    print(
        f"Wrote {path}\n"
        f"  {claimed} voxels claimed | {shared} claimed by more than one pair ({pct:.1f}%) "
        f"| max claims on one voxel: {peak}\n"
        f"  interscellar_meshes keeps the highest pair_id per voxel; overlap_count records the multiplicity"
    )

def _build_label_volume(records, shape):
    labels = np.zeros(shape, dtype=np.uint32)
    overlap_count = np.zeros(shape, dtype=np.uint16)

    usable = [
        r for r in records
        if r.get('pair_id') is not None and 'interscellar_mask' in r and r.get('interscellar_voxels')
    ]

    for record in usable:
        bbox, claim = record['union_bbox'], record['interscellar_mask']
        np.maximum(labels[bbox], claim.astype(np.uint32) * int(record['pair_id']), out=labels[bbox])
        overlap_count[bbox] += claim

    for record in usable:
        bbox, claim = record['union_bbox'], record['interscellar_mask']
        shared = int((overlap_count[bbox][claim] > 1).sum())
        record['shared_voxels'] = shared
        record['exclusive_voxels'] = int(record['interscellar_voxels']) - shared

    for record in records:
        record.setdefault('shared_voxels', 0)
        record.setdefault('exclusive_voxels', int(record.get('interscellar_voxels', 0)))

    return labels, overlap_count

def derive_output_stem(neighbor_pairs_path: str) -> str:
    base = os.path.splitext(os.path.basename(neighbor_pairs_path))[0]
    for marker in ("_neighbors_3d", "_neighbors", "neighbors"):
        base = base.replace(marker, "")
    return base.strip("_") or "interscellar"


def default_output_paths(neighbor_pairs_path: str, name_tag: str, output_dir=None):
    stem = derive_output_stem(neighbor_pairs_path)
    directory = output_dir or (os.path.dirname(neighbor_pairs_path) or ".")
    prefix = f"{stem}_{name_tag}" if name_tag else stem
    return (
        os.path.join(directory, f"{prefix}_volumes.csv"),
        os.path.join(directory, f"{prefix}_interscellar_volumes.zarr"),
        os.path.join(directory, f"{prefix}_cell_only_volumes.zarr"),
        os.path.join(directory, f"{prefix}_rejected_pairs.csv"),
    )

def export_pair_volumes(
    mask_3d,
    cell_a_id: int,
    cell_b_id: int,
    pair_id: int,
    out_dir: str,
    stem: str = "interscellar_adaptive",
    pad: int = 8,
    **kwargs: Any,
) -> Dict[str, Any]:

    import csv as _csv
    import os as _os
    import zarr

    result = compute_interscellar_volume_adaptive(
        mask_3d=mask_3d, cell_a_id=cell_a_id, cell_b_id=cell_b_id,
        pair_id=pair_id, return_debug=True, **kwargs,
    )
    if 'rejected' in result:
        raise ValueError(
            f"Pair {pair_id} ({cell_a_id}/{cell_b_id}) was rejected as {result['rejected']}: "
            f"surface separation {result['min_surface_separation_um']:.3f} um leaves no "
            f"connected background corridor reaching both cells. Pass "
            f"reject_unbridged=False to export it anyway."
        )
    bbox = result['union_bbox']
    crop = mask_3d[bbox]

    interscellar = result['interscellar_mask'].astype(np.uint32) * int(pair_id)
    cell_only = np.zeros(crop.shape, dtype=np.uint32)
    cell_only[(crop == cell_a_id) & ~result['adaptive_a_mask']] = int(cell_a_id)
    cell_only[(crop == cell_b_id) & ~result['adaptive_b_mask']] = int(cell_b_id)

    occupied = (interscellar > 0) | (cell_only > 0)
    coords = np.argwhere(occupied)
    lower = coords.min(axis=0)
    upper = coords.max(axis=0)
    tight = tuple(
        slice(max(0, int(lower[i]) - pad), min(crop.shape[i], int(upper[i]) + pad + 1))
        for i in range(3)
    )
    interscellar = interscellar[tight]
    cell_only = cell_only[tight]

    _os.makedirs(out_dir, exist_ok=True)
    chunks = tuple(min(64, n) for n in interscellar.shape)
    origin = [int(bbox[i].start + tight[i].start) for i in range(3)]

    for name, data, key in (
        (f"{stem}_interscellar_volumes.zarr", interscellar, "interscellar_meshes"),
        (f"{stem}_cell_only_volumes.zarr", cell_only, "labels"),
    ):
        store = zarr.open(_os.path.join(out_dir, name), mode="w")
        _create_array(store, key, data.shape, "uint32", chunks)[:] = data
        store.attrs["pair_id"] = int(pair_id)
        store.attrs["cell_a_id"] = int(cell_a_id)
        store.attrs["cell_b_id"] = int(cell_b_id)
        store.attrs["crop_origin_zyx"] = origin
        store.attrs["subtraction"] = "pairwise: each cell minus its own territory for this pair"

    with open(_os.path.join(out_dir, f"{stem}_volumes.csv"), "w", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerow(result)

    return {
        'pair_id': pair_id, 'cell_a_id': cell_a_id, 'cell_b_id': cell_b_id,
        'shape': interscellar.shape, 'crop_origin_zyx': origin,
        'interscellar_voxels': int((interscellar > 0).sum()),
        'cell_a_only_voxels': int((cell_only == cell_a_id).sum()),
        'cell_b_only_voxels': int((cell_only == cell_b_id).sum()),
        'territory_a_voxels': result['territory_a_voxels'],
        'territory_b_voxels': result['territory_b_voxels'],
        'volume_um3': result['interscellar_volume_um3'],
        'out_dir': out_dir,
    }

def main(argv=None) -> None:
    import argparse
    import csv
    import pickle
    import time

    parser = argparse.ArgumentParser(
        description="Adaptive 3D InterSCellar volume generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mask", required=True, help="3D segmentation label volume (.npy or .zarr)")
    parser.add_argument("--pairs", help="Neighbor pairs CSV or neighbor_graph .db; omit to run a single pair")
    parser.add_argument("--cell-a", type=int, help="Single-pair mode: first cell ID")
    parser.add_argument("--cell-b", type=int, help="Single-pair mode: second cell ID")
    parser.add_argument("--out", help="Per-pair volumes CSV. Auto-named from --pairs if omitted.")
    parser.add_argument("--out-zarr", help="Pair-labeled interscellar volume zarr. "
                                           "Auto-named from --pairs if omitted.")
    parser.add_argument("--name-tag", default="adaptive",
                        help="Tag distinguishing this pipeline's outputs from the absolute one. "
                             "Names follow <stem>_<tag>_volumes.csv / "
                             "<stem>_<tag>_interscellar_volumes.zarr")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for auto-named outputs. Defaults to the --pairs directory, "
                             "matching compute_interscellar_volumes_3d.")
    parser.add_argument(
        "--exclude-truncated",
        action="store_true",
        help="Skip pairs where either cell touches a face of the volume (recommended on a crop)",
    )
    parser.add_argument("--global-surface", help="Optional surfaces .pkl; must match the mask shape")
    parser.add_argument("--halo-bboxes", help="Optional halo bboxes .pkl; must lie inside the mask bounds")
    parser.add_argument("--max-distance-um", type=float, default=3.0)
    parser.add_argument("--surface-distance-um", type=float, default=0.5)
    parser.add_argument("--rho-threshold", type=float, default=0.5)
    parser.add_argument("--contact-rim-um", type=float, default=0.0)
    parser.add_argument("--max-inward-um", type=float, default=None)
    parser.add_argument("--z", type=float, default=0.56, help="Voxel size along z, um")
    parser.add_argument("--y", type=float, default=0.28, help="Voxel size along y, um")
    parser.add_argument("--x", type=float, default=0.28, help="Voxel size along x, um")
    parser.add_argument("--pair-id", type=int, default=None, help="Single-pair mode: label to stamp")
    parser.add_argument("--export-pairs",
                        help="Comma-separated pair_ids to export as per-pair interscellar + "
                             "pairwise cell-only volumes for visualize_pair_3d")
    parser.add_argument("--export-dir", default="pair_exports",
                        help="Directory for --export-pairs output (one subdirectory per pair)")
    parser.add_argument("--workers-load-volume", action="store_true",
                        help="Each worker loads the whole volume instead of reading pair crops "
                             "lazily. Faster when RAM allows n_jobs x volume; otherwise leave off.")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Pairs per chunk. Each chunk is folded into the outputs and released.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pair_ids already present in --out and append to the existing outputs")
    parser.add_argument("--pair-overlap-stats", action="store_true",
                        help="Deprecated and ignored: exact shared-voxel counts are now always "
                             "computed from the pair archive, at no memory cost.")
    parser.add_argument("--rejected-out", default=None,
                        help="CSV of pairs rejected as unbridged. Auto-named from --pairs "
                             "if omitted.")
    parser.add_argument("--no-reject-unbridged", dest="reject_unbridged",
                        action="store_false", default=True,
                        help="Keep the old behaviour: build a volume even when no connected "
                             "background corridor reaches both cells. Such a volume is two "
                             "disconnected intracellular lobes, so this is for comparison only.")
    parser.add_argument("--preview", choices=("both", "none"), default="both",
                        help="Write the dense interscellar_meshes/overlap_count preview arrays. "
                             "'none' keeps only the lossless pair archive, which the legacy "
                             "visualizers cannot read.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Worker processes for --pairs mode. Each loads its own copy of the volume.")
    parser.add_argument("--debug", action="store_true", help="Print rho ranges for a single pair")
    args = parser.parse_args(argv)

    if not args.pairs and (args.cell_a is None or args.cell_b is None):
        parser.error("provide --pairs, or both --cell-a and --cell-b")

    voxel_size_um = (args.z, args.y, args.x)

    if args.pairs:
        auto_csv, auto_zarr, _, auto_rejected = default_output_paths(
            args.pairs, args.name_tag, args.output_dir
        )
        if args.out is None:
            args.out = auto_csv
        if args.out_zarr is None and not args.export_pairs:
            args.out_zarr = auto_zarr
        if args.rejected_out is None and args.reject_unbridged and not args.export_pairs:
            args.rejected_out = auto_rejected
        print(f"Output CSV:  {args.out}")
        print(f"Output zarr: {args.out_zarr}")
        if args.rejected_out:
            print(f"Rejected:    {args.rejected_out}")
        for path in (args.out, args.out_zarr):
            if path and "Mobile Documents" in path:
                print(
                    "  WARNING: this path is inside iCloud Drive. Large zarr output there "
                    "will be synced and may stall; pass --output-dir to write locally."
                )
                break

    for path in (args.out, args.out_zarr, args.rejected_out):
        if not path:
            continue
        directory = os.path.dirname(os.path.abspath(path))
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            parser.error(f"cannot create the output directory {directory}: {exc}")
        if not os.access(directory, os.W_OK):
            parser.error(f"output directory is not writable: {directory}")

    mask_3d = _load_label_volume(args.mask)
    print(f"Loaded mask {mask_3d.shape} {mask_3d.dtype} from {args.mask}")

    global_surface = None
    if args.global_surface:
        with open(args.global_surface, "rb") as handle:
            payload = pickle.load(handle)
        global_surface = payload["global_surface"] if isinstance(payload, dict) else payload
        if global_surface.shape != mask_3d.shape:
            parser.error(
                f"--global-surface has shape {global_surface.shape} but the mask is {mask_3d.shape}. "
                "Surfaces computed on the full segmentation do not apply to a crop; omit the flag and "
                "they will be derived from the crop instead."
            )

    halo_bboxes = None
    if args.halo_bboxes:
        with open(args.halo_bboxes, "rb") as handle:
            payload = pickle.load(handle)
        halo_bboxes = payload.get("all_bboxes_with_halo", payload) if isinstance(payload, dict) else payload
        out_of_bounds = [
            cid for cid, bb in halo_bboxes.items()
            if any(bb[ax].stop > mask_3d.shape[ax] for ax in range(3))
        ]
        if out_of_bounds:
            parser.error(
                f"--halo-bboxes has {len(out_of_bounds)} boxes reaching outside the mask "
                f"{mask_3d.shape}; they were built on the full segmentation. Omit the flag and the "
                "bounding boxes will be derived from the crop instead."
            )

    geometry = dict(
        max_distance_um=args.max_distance_um,
        surface_distance_um=args.surface_distance_um,
        rho_threshold=args.rho_threshold,
        contact_rim_um=args.contact_rim_um,
        max_inward_um=args.max_inward_um,
        reject_unbridged=args.reject_unbridged,
    )

    if args.export_pairs:
        if not args.pairs:
            parser.error("--export-pairs needs --pairs to resolve pair_ids to cell IDs")
        wanted = [int(p) for p in args.export_pairs.replace(" ", "").split(",") if p]
        lookup = {pid: (a, b) for a, b, pid in _load_pairs(args.pairs)}
        missing = [p for p in wanted if p not in lookup]
        if missing:
            parser.error(f"pair_ids not present in {args.pairs}: {missing}")

        export_stem = f"{derive_output_stem(args.pairs)}_{args.name_tag}".strip("_")
        print("Computing halo bounding boxes (one pass over the volume)...")
        involved = {c for pid in wanted for c in lookup[pid]}
        boxes = compute_halo_bboxes(mask_3d, voxel_size_um, args.max_distance_um,
                                    cell_ids=involved)

        exported = 0
        for pid in wanted:
            cell_a_id, cell_b_id = lookup[pid]
            out_dir = os.path.join(args.export_dir, f"pair_{pid}")
            try:
                info = export_pair_volumes(
                    mask_3d, cell_a_id, cell_b_id, pid, out_dir,
                    stem=export_stem,
                    pad=8,
                    voxel_size_um=voxel_size_um,
                    global_surface=global_surface,
                    halo_bboxes=boxes,
                    **geometry,
                )
            except ValueError as exc:
                # An unbridged pair is an expected outcome here, not a crash.
                print(f"pair {pid} (cells {cell_a_id}/{cell_b_id}) -> skipped\n  {exc}")
                continue
            exported += 1
            print(
                f"pair {pid} (cells {cell_a_id}/{cell_b_id}) -> {out_dir}\n"
                f"  crop {info['shape']} at z,y,x {info['crop_origin_zyx']} | "
                f"interscellar {info['interscellar_voxels']} vox = {info['volume_um3']:.1f} um^3\n"
                f"  cell-only A {info['cell_a_only_voxels']} vox (minus T_A {info['territory_a_voxels']}), "
                f"B {info['cell_b_only_voxels']} vox (minus T_B {info['territory_b_voxels']})"
            )
        if not exported:
            parser.error(
                f"none of the requested pairs could be exported: all {len(wanted)} were "
                f"rejected as unbridged. Pass --no-reject-unbridged to export them anyway."
            )
        print(f"\nDone ({exported}/{len(wanted)} exported). Visualize with:\n"
              f"  visualize-pair-3d --pair-id <ID> \\\n"
              f"    --cell-only-zarr {args.export_dir}/pair_<ID>/{export_stem}_cell_only_volumes.zarr \\\n"
              f"    --interscellar-zarr {args.export_dir}/pair_<ID>/{export_stem}_interscellar_volumes.zarr")
        return

    if args.pairs:
        pairs = _load_pairs(args.pairs)
        print(f"Loaded {len(pairs)} pairs from {args.pairs}")

        if halo_bboxes is None:
            print("Computing halo bounding boxes (one pass over the volume)...")
            halo_bboxes = compute_halo_bboxes(mask_3d, voxel_size_um, args.max_distance_um)
        present = set(halo_bboxes)
        present.discard(0)
        edge_ids = truncated_cell_ids(mask_3d) if args.exclude_truncated else set()
        if args.exclude_truncated:
            print(f"Cells touching a volume face: {len(edge_ids)}")

        runnable, absent, truncated = [], 0, 0
        for cell_a_id, cell_b_id, pair_id in pairs:
            if cell_a_id not in present or cell_b_id not in present:
                absent += 1
            elif cell_a_id in edge_ids or cell_b_id in edge_ids:
                truncated += 1
            else:
                runnable.append((cell_a_id, cell_b_id, pair_id))
        print(
            f"Cells present in this volume: {len(present)} | pairs to compute: {len(runnable)} "
            f"(skipped {absent} not in volume, {truncated} truncated)"
        )

        if not runnable:
            parser.error(
                "no pairs are computable in this volume -- check that the label IDs in "
                "--pairs match the label IDs in --mask"
            )

        paired_cells = {c for pair in runnable for c in pair[:2]}
        halo_bboxes = {c: b for c, b in halo_bboxes.items() if c in paired_cells}

        worker_kwargs = dict(
            voxel_size_um=voxel_size_um,
            global_surface=global_surface,
            halo_bboxes=halo_bboxes,
            return_debug=False,
            **geometry,
        )

        need_masks = bool(args.out_zarr)
        mask_shape = mask_3d.shape
        volume_gib = mask_3d.nbytes / 2**30

        lazy = (
            args.n_jobs > 1
            and not args.workers_load_volume
            and _open_label_volume_lazy(args.mask) is not None
        )
        if args.n_jobs > 1:
            if lazy:
                del mask_3d
                mask_3d = None
                import gc
                gc.collect()
                print(
                    f"Volume {mask_shape} = {volume_gib:.2f} GiB. Workers read pair crops "
                    f"lazily from the store, so {args.n_jobs} workers add little beyond it."
                )
            else:
                print(
                    f"Volume {mask_shape} = {volume_gib:.2f} GiB per worker; "
                    f"{args.n_jobs} workers need ~{volume_gib * args.n_jobs:.1f} GiB resident. "
                    f"Lower --n-jobs if that exceeds your RAM."
                )

        if args.resume and args.out and os.path.exists(args.out):
            with open(args.out, newline="") as handle:
                done = {int(row['pair_id']) for row in csv.DictReader(handle) if row.get('pair_id')}
            before = len(runnable)
            runnable = [p for p in runnable if p[2] not in done]
            print(f"Resuming: {len(done)} pairs already recorded, {len(runnable)} of {before} left")

        appending = bool(args.resume and args.out and os.path.exists(args.out))
        csv_handle = csv_writer = None
        if args.out:
            csv_handle = open(args.out, "a" if appending else "w", newline="")
            csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_COLUMNS, extrasaction='ignore')
            if not appending:
                csv_writer.writeheader()

        rejected_path = args.rejected_out
        rejected_handle = rejected_writer = None
        if rejected_path:
            rejected_handle = open(rejected_path, "a" if appending else "w", newline="")
            rejected_writer = csv.DictWriter(
                rejected_handle, fieldnames=REJECTED_COLUMNS, extrasaction='ignore'
            )
            if not appending:
                rejected_writer.writeheader()

        store = ds_labels = ds_overlap = archive = None
        written_ids = set()
        if args.out_zarr:
            store, ds_labels, ds_overlap, written_ids, archive = _open_output_zarr(
                args.out_zarr, mask_shape, appending, voxel_size_um, geometry,
                preview=(args.preview != "none"),
            )
            if appending and not written_ids:
                print(
                    "  note: --out-zarr has no record of previously written pairs, so pairs "
                    "already listed in the CSV will not be added to it"
                )

        chunk_size = max(1, args.chunk_size)
        n_done = n_failed = n_rejected = n_direct = n_near = 0
        total_um3 = 0.0
        started = time.time()
        try:
            for offset in range(0, len(runnable), chunk_size):
                chunk = runnable[offset:offset + chunk_size]
                if args.n_jobs > 1:
                    chunk_records = _run_pairs_parallel(
                        args.mask, chunk, args.n_jobs, need_masks, worker_kwargs,
                        verbose=False, lazy=lazy,
                    )
                else:
                    chunk_records, chunk_dropped = compute_interscellar_volumes_adaptive_for_pairs(
                        mask_3d, chunk, keep_masks=need_masks, verbose=False, **worker_kwargs
                    )
                    chunk_records = chunk_records + chunk_dropped['rejected']
                failures = [r for r in chunk_records if 'error' in r]
                n_failed += len(failures)
                rejects = [r for r in chunk_records if 'rejected' in r]
                chunk_records = [
                    r for r in chunk_records if 'error' not in r and 'rejected' not in r
                ]

                if rejects:
                    n_rejected += len(rejects)
                    if rejected_writer is not None:
                        rejected_writer.writerows(rejects)
                        rejected_handle.flush()

                fresh = [r for r in chunk_records
                         if r.get('interscellar_voxels') and int(r['pair_id']) not in written_ids]
                if ds_labels is not None:
                    for record in fresh:
                        _write_pair_into(ds_labels, ds_overlap, record)
                if archive is not None and fresh:
                    archive.append(fresh)
                written_ids.update(int(r['pair_id']) for r in fresh)

                for record in chunk_records:
                    total_um3 += record.get('total_interscellar_volume_um3', 0.0)
                    if record.get('interface_kind') == INTERFACE_KIND['direct']:
                        n_direct += 1
                    else:
                        n_near += 1
                if csv_writer is not None:
                    csv_writer.writerows(chunk_records)
                    csv_handle.flush()

                n_done += len(chunk_records)
                processed = n_done + n_rejected + n_failed
                rate = processed / max(time.time() - started, 1e-6)
                remaining = (len(runnable) - processed) / rate if rate else 0
                print(
                    f"  {processed}/{len(runnable)} pairs | {n_done} accepted, "
                    f"{n_rejected} unbridged | {rate:.1f} pairs/s | "
                    f"~{remaining/60:.1f} min left | running total {total_um3:.1f} um^3",
                    flush=True,
                )
        finally:
            if csv_handle is not None:
                csv_handle.close()
            if rejected_handle is not None:
                rejected_handle.close()

        if archive is not None and ds_overlap is not None:
            print("Recording per-pair shared voxels against the finished volume...")
            archive.fill_shared_voxels(ds_overlap)
            patch_csv_shared_voxels(args.out, archive)

        print(
            f"\nSummary: {len(runnable)} requested | {n_direct} accepted-direct, "
            f"{n_near} accepted-near | {n_rejected} rejected-unbridged | "
            f"{absent} absent, {truncated} truncated, {n_failed} failed"
        )
        if n_rejected and args.reject_unbridged:
            pct = 100.0 * n_rejected / max(1, len(runnable))
            print(
                f"  {n_rejected} pairs ({pct:.1f}%) had no connected background corridor "
                f"reaching both cells at --max-distance-um {args.max_distance_um}. A high "
                f"share here usually means the neighbor list contains pairs far beyond the "
                f"threshold; see {rejected_path or 'the rejected-pairs CSV'}."
            )
        if args.out:
            print(f"Wrote {n_done} rows to {args.out} (total {total_um3:.2f} um^3)")
        if archive is not None:
            print(f"Wrote {archive.n_rows} pair footprints to {args.out_zarr}/{_ARCHIVE_GROUP}")
        if ds_labels is not None:
            _report_overlap(ds_labels, ds_overlap, args.out_zarr)
        return

    else:
        records = [
            compute_interscellar_volume_adaptive(
                mask_3d=mask_3d,
                cell_a_id=args.cell_a,
                cell_b_id=args.cell_b,
                voxel_size_um=voxel_size_um,
                global_surface=global_surface,
                halo_bboxes=halo_bboxes,
                pair_id=args.pair_id,
                return_debug=args.debug or bool(args.out_zarr),
                **geometry,
            )
        ]
        r = records[0]
        if 'rejected' in r:
            print(
                f"Pair {args.cell_a}-{args.cell_b}: REJECTED ({r['rejected']})\n"
                f"  surface separation {r['min_surface_separation_um']:.3f} um vs "
                f"--max-distance-um {args.max_distance_um}; "
                f"{r['n_candidate_components']} candidate corridor component(s), "
                f"another cell in the way: {r['blocked_by_other_cell']}\n"
                f"  No connected background corridor reaches both cells, so a volume here "
                f"would be two disconnected intracellular lobes. Pass "
                f"--no-reject-unbridged to build it anyway."
            )
            return
        print(
            f"Pair {args.cell_a}-{args.cell_b}: {r['interscellar_voxels']} voxels "
            f"({r['interscellar_volume_um3']:.3f} um^3) | "
            f"territory_a={r['territory_a_voxels']} corridor={r['corridor_voxels']} "
            f"territory_b={r['territory_b_voxels']} | "
            f"{'direct' if r['is_direct_contact'] else 'near'}, "
            f"{r['n_components']} component(s)"
        )
        if args.debug:
            print(f"  rho_a range: {np.nanmin(r['rho_a']):.3f} to {np.nanmax(r['rho_a']):.3f}")
            print(f"  rho_b range: {np.nanmin(r['rho_b']):.3f} to {np.nanmax(r['rho_b']):.3f}")

    labels = overlap_count = None
    if any('interscellar_mask' in r for r in records):
        labels, overlap_count = _build_label_volume(records, mask_3d.shape)

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        total = sum(r['total_interscellar_volume_um3'] for r in records)
        print(f"Wrote {len(records)} rows to {args.out} (total {total:.2f} um^3)")

    if args.out_zarr:
        import zarr

        if labels is None:
            parser.error("--out-zarr needs the pair masks; rerun without --n-jobs stripping them")

        claimed = int((overlap_count > 0).sum())
        shared = int((overlap_count > 1).sum())
        store = zarr.open(args.out_zarr, mode="w")
        chunks = tuple(min(c, s_) for c, s_ in zip((64, 256, 256), labels.shape))
        _create_array(store, "interscellar_meshes", labels.shape, "uint32", chunks)[:] = labels
        _create_array(store, "overlap_count", overlap_count.shape, "uint16", chunks)[:] = overlap_count
        store.attrs["description"] = "Adaptive interscellar volumes labeled by pair ID"
        store.attrs["label_collision_policy"] = "highest pair_id wins (np.maximum), matching compute_interscellar_volumes_3d"
        store.attrs["overlap_count_meaning"] = "number of pairs claiming each voxel; >1 means the pair ID shown is only one of several"
        store.attrs["num_pairs"] = len(records)
        store.attrs["voxel_size_um"] = list(voxel_size_um)
        store.attrs["coordinate_system"] = "same_as_input_segmentation"
        store.attrs["axes"] = ["z", "y", "x"]
        store.attrs.update({k: v for k, v in geometry.items() if v is not None})
        pct = 100.0 * shared / claimed if claimed else 0.0
        print(
            f"Wrote {args.out_zarr}\n"
            f"  {claimed} voxels claimed | {shared} claimed by more than one pair ({pct:.1f}%) "
            f"| max claims on one voxel: {int(overlap_count.max()) if claimed else 0}\n"
            f"  interscellar_meshes keeps the highest pair_id per voxel; overlap_count records the multiplicity"
        )

if __name__ == "__main__":
    main()
