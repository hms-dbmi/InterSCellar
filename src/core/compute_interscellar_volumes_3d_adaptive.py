"""Adaptive 3D InterSCellar volume generator.

This module is self-contained by design: it leaves the behavior of
``compute_interscellar_volumes_3d`` untouched and builds a parallel, pair-local
adaptive interaction mask that reuses the same crop conventions, cell-surface
masks, and intercellular corridor definition used by the package.

Definition
----------
For a neighboring pair A-B the interscellar volume is a single union of three
disjoint parts::

    V(A, B) = T_A  u  G_AB  u  T_B

where ``G_AB`` is the extracellular corridor between the two cells and ``T_A``,
``T_B`` are the pair-facing intracellular territories.

``G_AB`` (corridor) is the set of background voxels whose summed Euclidean
distance to the two cell surfaces is within ``max_distance_um``::

    G_AB = { x not in A u B : d_dA(x) + d_dB(x) <= max_distance_um }

``T_A`` (territory) is defined by a continuous normalized interaction depth. The
seed is the whole interaction zone -- the pair-facing part of A's boundary
``dA_c`` together with the corridor voxels ``G_AB`` -- competed against the
remaining boundary ``dA_nc``::

    rho(x) = d_c(x) / (d_c(x) + d_o(x)),    T_A = { x in A : rho(x) <= rho_threshold }

where d_c is the distance to ``dA_c u G_AB`` and d_o the distance to ``dA_nc``,
with default ``rho_threshold=0.5``.

Seeding the inward field on the corridor as well as the boundary is what makes the
territory follow the detected extracellular gap. Where the labels abut there is no
corridor and the boundary carries the seed; where they are separated the corridor
does.

The default ``rho_threshold=0.5`` is the geometric bisector: a voxel is included
when it is closer to the interaction zone than to the non-contact cell surface. It
is not a fixed biological distance. This is intentionally shape-aware and should be
interpreted as a geometric, not pharmacologic, threshold. Use ``max_inward_um`` to
impose an absolute penetration cap.

Every pair supplied is computed. The pair-facing boundary region is built from
inclusive rules only -- adjacency to the partner cell, plus every boundary voxel
within ``surface_distance_um`` of the pair's own closest approach -- so no threshold
can leave a pair without an intracellular territory. Surface separation is reported
for reference, never used to reject a pair.

All distances are Euclidean in physical units, using anisotropic voxel sampling
throughout, so the same threshold means the same physical distance along z as
along x and y.

Pair IDs are unchanged, and the final pair-local mask can be returned either as
a binary mask or as a pair-labeled mask, matching the rest of the project.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, generate_binary_structure

# 26-connectivity, used for the surface fallback so it matches ``global_surface_26n`` in
# find_cell_neighbors_3d. The neighbor detection step measures surface-to-surface distance
# on a 26-connected surface, so deriving a thinner 6-connected surface here would make
# ``surface_distance_um`` mean something subtly different from the neighbor threshold.
_CONN_26 = generate_binary_structure(3, 3)


def _pair_union_bbox(
    halo_bboxes: Dict[int, Tuple[slice, slice, slice]],
    cell_a_id: int,
    cell_b_id: int,
) -> Tuple[slice, slice, slice]:
    """Return the union of the two pair-local halo bounding boxes."""
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
    """Minimal bbox around both cells, padded by the max-distance halo on each axis."""
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
    """Boundary voxels of ``mask``, from the global surface when available."""
    if global_surface_crop is not None:
        return mask & global_surface_crop
    return mask & ~binary_erosion(mask, structure=_CONN_26)


def _one_contact_region(
    surface: np.ndarray,
    mask_other: np.ndarray,
    dist_to_other_surface: np.ndarray,
    surface_distance_um: float,
) -> np.ndarray:
    """The part of one cell's boundary that faces its partner.

    Union of two inclusive rules, never a gate:

    1. boundary directly adjacent to the partner cell, where the labels abut;
    2. boundary within ``surface_distance_um`` of the pair's *closest approach*.

    Rule 2 is measured relative to the minimum surface-to-surface distance rather than
    from zero, so it always selects a real patch of facing boundary no matter how far
    apart the pair is. That is what guarantees every pair from the neighbor list an
    intracellular territory: separation can shrink the region but never empty it.

    The corridor is deliberately *not* used here. It reaches up to ``max_distance_um``
    around the rim of the contact, so admitting every boundary voxel adjacent to it
    would claim roughly half of a touching cell's boundary. That would leave the
    competing set ``dA_nc`` reduced to the far cap, ``d_o`` would degenerate into a
    distance-to-the-far-pole field, and the rho bisector would flatten into a plane at
    constant depth instead of following the local geometry. The corridor belongs in the
    territory *seed* (see ``_cell_territory``), which is a separate role.
    """
    if not surface.any():
        return np.zeros_like(surface, dtype=bool)

    contact = surface & binary_dilation(mask_other, structure=_CONN_26)

    nearest = float(dist_to_other_surface[surface].min())
    contact |= surface & (dist_to_other_surface <= nearest + surface_distance_um)

    return contact


def _pair_distance_fields(
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance in micrometers from every voxel of the crop to each cell's surface.

    Computed once per pair and reused for the corridor, the contact regions and the
    measured separation. Previously the corridor and the contact step each ran their own
    pair of transforms, and the corridor additionally ran two eight-iteration binary
    dilations to bound its work -- all of which this replaces.
    """
    d_a = distance_transform_edt(~surface_a, sampling=voxel_size_um).astype(np.float32)
    d_b = distance_transform_edt(~surface_b, sampling=voxel_size_um).astype(np.float32)
    return d_a, d_b


def _corridor_from_fields(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    d_a: np.ndarray,
    d_b: np.ndarray,
    max_distance_um: float,
) -> np.ndarray:
    """Extracellular gap: background voxels with d_A + d_B <= max_distance_um.

    The criterion is applied directly to the summed distance field. The old
    dilation-derived candidate region was a city-block ball, which clipped voxels that
    satisfy the Euclidean criterion diagonally; testing the field itself is both cheaper
    and exactly the definition.
    """
    return ~(mask_a | mask_b) & ((d_a + d_b) <= max_distance_um)


def _contact_regions(
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    d_a: np.ndarray,
    d_b: np.ndarray,
    surface_distance_um: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pair-facing boundary region on each cell, plus the measured surface separation."""
    if not surface_a.any() or not surface_b.any():
        empty = np.zeros_like(surface_a, dtype=bool)
        return empty, empty.copy(), float('inf')

    separation = float(d_a[surface_b].min())
    contact_a = _one_contact_region(surface_a, mask_b, d_b, surface_distance_um)
    contact_b = _one_contact_region(surface_b, mask_a, d_a, surface_distance_um)
    return contact_a, contact_b, separation


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
    """Adaptive intracellular territory facing the partner cell.

    The inward distance field grows from the whole interaction zone -- the pair-facing
    boundary region *and* the extracellular corridor voxels between the cells -- not
    from a membrane patch alone. Where the labels abut there is no corridor and the
    boundary carries the seed; where they are separated the corridor does, so a pair
    that never touches still gets a territory.

    Returns ``(territory, rho, d_contact)``. ``rho`` is NaN outside the cell.
    """
    empty = np.zeros_like(cell_mask, dtype=bool)
    rho = np.full(cell_mask.shape, np.nan, dtype=np.float32)

    seed = contact | corridor
    if not cell_mask.any() or not seed.any():
        return empty, rho, np.full(cell_mask.shape, np.inf, dtype=np.float32)

    d_contact = distance_transform_edt(~seed, sampling=voxel_size_um).astype(np.float32)

    # The rim is an exclusion zone in physical units around the interaction zone. Without
    # it the non-contact surface abuts the contact region's perimeter, d_other -> 0 there,
    # and the territory pinches shut at the edge.
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
        # Fully engulfed: no competing boundary, so d_other -> inf and rho -> 0
        # everywhere inside the cell. This is the limit of the formula above.
        rho[inside] = 0.0

    territory = np.zeros_like(cell_mask, dtype=bool)
    territory[inside] = rho[inside] <= rho_threshold
    if max_inward_um is not None:
        territory[inside] &= d_contact[inside] <= max_inward_um

    return territory, rho, d_contact


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
    pair_id: Optional[int] = None,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """Compute a pairwise adaptive InterSCellar interaction volume.

    Parameters
    ----------
    mask_3d:
        Full 3D label volume with cell IDs and 0 for background.
    cell_a_id, cell_b_id:
        Cell IDs for the queried pair.
    voxel_size_um:
        Physical voxel spacing in z, y, x order.
    global_surface:
        Optional precomputed global surface mask. If not provided, the function
        derives a 26-connected surface mask from the cell segmentation, matching
        ``global_surface_26n`` in find_cell_neighbors_3d.
    halo_bboxes:
        Optional mapping from cell ID to the halo-expanded bbox used by the
        package for pair-local processing. If omitted, a conservative box is
        derived from the cell masks.
    max_distance_um:
        Cutoff on d_A + d_B for the extracellular corridor, in micrometers.
    surface_distance_um:
        Optional widening of the pair-facing boundary region, in micrometers. Purely
        additive: boundary voxels within this distance of the partner's surface are
        included on top of the adjacency rules. It cannot exclude anything, and a pair
        that meets it nowhere still gets a territory.
    rho_threshold:
        Normalized geometric threshold for the adaptive intracellular territory.
        The default 0.5 is the bisector between the contact surface and the
        non-contact cell surface.
    contact_rim_um:
        Width of a rim around the interaction zone excluded from the non-contact
        surface, preventing the territory from pinching shut at the contact edge.
    max_inward_um:
        Optional absolute cap on inward penetration, as a physical distance from
        the contact surface. Disabled by default.
    exclude_other_cells:
        Remove voxels belonging to third-party cells from the corridor, so a pair
        volume never claims a neighbor's interior.
    pair_id:
        QP pair ID used for labeling convention compatibility.
    return_debug:
        If True, also return the component masks and the rho / distance fields.
    """
    if cell_a_id == cell_b_id:
        raise ValueError("cell_a_id and cell_b_id must be different for a pair computation")

    if halo_bboxes is None:
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

    # 1. One pair of surface distance fields, reused by everything below.
    if surface_a.any() and surface_b.any():
        d_a, d_b = _pair_distance_fields(surface_a, surface_b, voxel_size_um)
    else:
        inf = np.full(mask_crop.shape, np.inf, dtype=np.float32)
        d_a, d_b = inf, inf

    # 2. Extracellular corridor between the two cells.
    corridor = _corridor_from_fields(mask_a, mask_b, d_a, d_b, max_distance_um)
    if exclude_other_cells:
        corridor &= ~((mask_crop != 0) & ~mask_a & ~mask_b)

    # 3. Pair-facing boundary region on each cell. Inclusive, never a gate: every pair
    #    from the neighbor list gets a region, and so a territory.
    contact_a, contact_b, separation_um = _contact_regions(
        surface_a, surface_b, mask_a, mask_b, d_a, d_b, surface_distance_um
    )

    # 4. Adaptive intracellular territory, grown inward from the boundary region and the
    #    corridor together.
    territory_a, rho_a, d_contact_a = _cell_territory(
        mask_a, surface_a, contact_a, corridor,
        voxel_size_um, rho_threshold, contact_rim_um, max_inward_um,
    )
    territory_b, rho_b, d_contact_b = _cell_territory(
        mask_b, surface_b, contact_b, corridor,
        voxel_size_um, rho_threshold, contact_rim_um, max_inward_um,
    )

    # 5. Union. The three parts are disjoint by construction (territories lie inside
    #    their own cell, the corridor lies in background), so the voxel count of the
    #    union is the total and nothing is counted twice.
    interscellar_mask = territory_a | corridor | territory_b
    interscellar_voxels = int(interscellar_mask.sum())
    voxel_volume_um3 = float(np.prod(voxel_size_um))
    interscellar_volume_um3 = interscellar_voxels * voxel_volume_um3

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
        # Component breakdown (disjoint, so these sum to the total).
        'corridor_voxels': int(corridor.sum()),
        'territory_a_voxels': int(territory_a.sum()),
        'territory_b_voxels': int(territory_b.sum()),
        # Contact diagnostics, reported for reference only -- none of these gates the
        # computation. min_surface_separation_um is the same quantity
        # find_cell_neighbors_3d records per pair, so the two can be compared.
        'min_surface_separation_um': separation_um,
        'contact_a_voxels': int(contact_a.sum()),
        'contact_b_voxels': int(contact_b.sum()),
        'has_contact_patch': bool(contact_a.any() and contact_b.any()),
        'adaptive_territory_valid': bool(territory_a.any() and territory_b.any()),
        # Parameters used.
        'voxel_volume_um3': voxel_volume_um3,
        'max_distance_threshold_um': max_distance_um,
        'surface_distance_um': surface_distance_um,
        'rho_threshold': rho_threshold,
        'contact_rim_um': contact_rim_um,
        'max_inward_um': max_inward_um,
    }

    if pair_id is not None:
        result['labeled_interscellar_mask'] = interscellar_mask.astype(np.uint32) * int(pair_id)
        result['labeled_pair_id'] = int(pair_id)

    if return_debug:
        result.update({
            'corridor_mask': corridor,
            'adaptive_a_mask': territory_a,
            'adaptive_b_mask': territory_b,
            'contact_surface_a': contact_a,
            'contact_surface_b': contact_b,
            'rho_a': rho_a,
            'rho_b': rho_b,
            'd_contact_a': d_contact_a,
            'd_contact_b': d_contact_b,
        })

    return result



# Pair-level API alias, kept for compatibility with the package's naming convention.
compute_interscellar_volume_adaptive_for_pair = compute_interscellar_volume_adaptive


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #

def truncated_cell_ids(mask_3d: np.ndarray) -> set:
    """Cell IDs touching any face of the volume.

    A cell clipped by the crop boundary has a flat, fabricated surface there. That
    surface is counted as non-contact boundary, which biases rho, and it can seed a
    spurious corridor. Such cells are usually excluded from a cropped analysis.
    """
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
    """Run the adaptive computation over many pairs, skipping ones this volume cannot support.

    ``pairs`` is an iterable of ``(cell_a_id, cell_b_id, pair_id)``. Pairs naming a
    cell absent from ``mask_3d`` are skipped, which is the normal case when the pair
    list was built on a full segmentation and the volume here is a crop of it.

    Returns ``(records, skipped)`` where ``records`` is a list of per-pair result dicts
    and ``skipped`` maps a reason to the pairs dropped. With ``keep_masks=False`` the
    array fields are dropped from each record, which matters on a whole-volume run where
    holding one crop per pair would otherwise dominate memory.
    """
    present = set(np.unique(mask_3d).tolist())
    present.discard(0)

    dropped: Dict[str, list] = {'cell_absent': [], 'truncated': [], 'failed': []}
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

        if not keep_masks:
            result = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        records.append(result)

    if verbose:
        print(
            f"Computed {len(records)} pairs | skipped: "
            f"{len(dropped['cell_absent'])} not in volume, "
            f"{len(dropped['truncated'])} truncated, "
            f"{len(dropped['failed'])} failed"
        )
    return records, dropped


def _open_label_volume_lazy(path: str):
    """Return a lazily-indexable 3D label volume, or None if it must be read whole.

    Workers only ever touch ``mask_3d[union_bbox]`` -- one pair-local crop at a time --
    so on a whole-segmentation run there is no reason for each of them to hold a full
    copy. A zarr array (or a memory-mapped .npy) slices to exactly the requested region,
    which turns per-worker memory from the size of the volume into the size of a crop.
    """
    if path.endswith(".npy"):
        array = np.load(path, mmap_mode="r")
        return array if array.ndim == 3 else None

    import zarr

    group = zarr.open(path, mode="r")
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
    if array is None or getattr(array, "ndim", 0) != 3:
        return None
    return array


def _load_label_volume(path: str) -> np.ndarray:
    """Load a 3D label volume from .npy or an OME-Zarr store."""
    if path.endswith(".npy"):
        return np.load(path)

    import zarr

    group = zarr.open(path, mode="r")
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
        array = array.astype(array.dtype.newbyteorder("="))
    return array


def _load_pairs(path: str):
    """Read (cell_a_id, cell_b_id, pair_id) triples from a neighbor CSV or .db.

    Deliberately self-contained -- no import from the rest of the package -- so this
    file can be run as a plain script without importing ``interscellar`` and its
    optional dependencies.
    """
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


# --------------------------------------------------------------------------- #
# Parallel execution. Each worker loads the label volume once in an initializer,
# so the array is never pickled per task.
# --------------------------------------------------------------------------- #

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
        if keep_mask:
            # bit-pack so only ~1/8 of the mask crosses the process boundary
            slim['_packed'] = np.packbits(result['interscellar_mask'])
            slim['_shape'] = result['interscellar_mask'].shape
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

    for record in records:
        if '_packed' in record:
            shape = record.pop('_shape')
            packed = record.pop('_packed')
            record['interscellar_mask'] = np.unpackbits(
                packed, count=int(np.prod(shape))
            ).astype(bool).reshape(shape)
    return records


def compute_halo_bboxes(
    mask_3d: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    max_distance_um: float,
    cell_ids=None,
) -> Dict[int, Tuple[slice, slice, slice]]:
    """Per-cell bounding boxes padded by the interaction halo, from one pass.

    Essential for whole-volume runs. Without a bbox table, every pair falls back to
    ``_fallback_union_bbox``, which scans the entire label volume twice to locate its two
    cells -- O(pairs x volume). One ``find_objects`` pass replaces all of it.

    The padding matches ``_fallback_union_bbox`` so results are identical either way.
    """
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
    'min_surface_separation_um', 'contact_a_voxels', 'contact_b_voxels',
    'has_contact_patch', 'adaptive_territory_valid',
    'shared_voxels', 'exclusive_voxels', 'voxel_volume_um3',
    'max_distance_threshold_um', 'surface_distance_um', 'rho_threshold',
    'contact_rim_um', 'max_inward_um',
]


def _create_array(store, name, shape, dtype, chunks, fill_value=0):
    """Create an array in a zarr group across zarr 2 and 3.

    zarr 3 removed ``Group.create_dataset`` in favour of ``Group.create_array``.
    """
    if hasattr(store, "create_array"):          # zarr 3
        try:
            return store.create_array(name, shape=shape, dtype=dtype, chunks=chunks,
                                      fill_value=fill_value)
        except TypeError:
            return store.create_array(name, shape=shape, dtype=dtype, chunks=chunks)
    return store.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks,   # zarr 2
                                fill_value=fill_value)


def _open_output_zarr(path, shape, appending, voxel_size_um, geometry):
    """Create (or reopen for resume) the pair-label and overlap-count datasets.

    Both are allocated on disk at full volume shape and written region by region, so a
    whole-segmentation run never holds them in RAM.
    """
    import zarr

    store = zarr.open(path, mode="a" if appending else "w")
    if "interscellar_meshes" not in store:
        chunks = (min(32, shape[0]), min(256, shape[1]), min(256, shape[2]))
        _create_array(store, "interscellar_meshes", shape, "uint32", chunks)
        _create_array(store, "overlap_count", shape, "uint16", chunks)
    store.attrs["description"] = "Adaptive interscellar volumes labeled by pair ID"
    store.attrs["contains"] = "interscellar volumes only; cells are not written"
    store.attrs["label_collision_policy"] = (
        "highest pair_id wins (np.maximum), matching compute_interscellar_volumes_3d"
    )
    store.attrs["overlap_count_meaning"] = (
        "number of pairs claiming each voxel; >1 means the pair ID shown is one of several"
    )
    store.attrs["voxel_size_um"] = list(voxel_size_um)
    store.attrs["coordinate_system"] = "same_as_input_segmentation"
    store.attrs["axes"] = ["z", "y", "x"]
    # Resuming into outputs built with different thresholds would silently blend two
    # parameter sets into one volume. Refuse rather than corrupt.
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

    # interscellar_meshes is idempotent under np.maximum, but overlap_count accumulates,
    # so re-writing a pair would inflate it. Track what has been folded in already.
    written = set(store.attrs.get("written_pair_ids", [])) if appending else set()
    if not appending:
        store.attrs["written_pair_ids"] = []
    return store, store["interscellar_meshes"], store["overlap_count"], written


def _write_pair_into(ds_labels, ds_overlap, record):
    """Fold one pair's mask into the on-disk label and overlap volumes."""
    bbox = record['union_bbox']
    claim = record['interscellar_mask']
    pair_id = int(record['pair_id'])

    region = np.asarray(ds_labels[bbox])
    np.maximum(region, claim.astype(np.uint32) * pair_id, out=region)
    ds_labels[bbox] = region

    counts = np.asarray(ds_overlap[bbox])
    counts += claim
    ds_overlap[bbox] = counts


def _report_overlap(ds_labels, ds_overlap, path, slab=16):
    """Global overlap summary, read back a slab at a time."""
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
    """Rasterize every pair mask into one label volume, and account for overlaps.

    Pairs that share a cell routinely claim the same voxels -- each pair's territory is
    computed independently against the full segmentation, so a cell in five pairs
    contributes territory to all five, and those territories overlap near the shared
    membrane. Per-pair volumes in the CSV are unaffected by this: each counts all of its
    own voxels. Only a single-label raster has to pick one winner per voxel.

    Collision policy is ``np.maximum`` -- the highest pair ID wins -- which matches
    ``compute_interscellar_volumes_3d`` and, unlike first-writer-wins, does not depend on
    the order results arrive in. That matters because ``--n-jobs`` completes batches out
    of order.

    ``overlap_count`` records how many pairs claimed each voxel, so what the single label
    hides stays recoverable. Each record also gains ``shared_voxels`` and
    ``exclusive_voxels``.
    """
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
    """Stem for output names, using the same rule as ``compute_interscellar_volumes_3d``.

    That pipeline builds every output name from the neighbor-pairs file, stripping the
    neighbour markers off the basename (see ``wrapper_3d.compute_interscellar_volumes_3d``).
    Mirroring it keeps the two pipelines' outputs sitting side by side and lets
    ``visualize_pair_3d`` resolve a pair_id from ``{stem}_volumes.csv`` unchanged.
    """
    base = os.path.splitext(os.path.basename(neighbor_pairs_path))[0]
    for marker in ("_neighbors_3d", "_neighbors", "neighbors"):
        base = base.replace(marker, "")
    return base.strip("_") or "interscellar"


def default_output_paths(neighbor_pairs_path: str, name_tag: str, output_dir=None):
    """(csv, interscellar_zarr, cell_only_zarr) following the package naming convention.

    ``name_tag`` separates the two pipelines: ``adaptive`` here, ``absolute`` for
    ``compute_interscellar_volumes_3d``.
    """
    stem = derive_output_stem(neighbor_pairs_path)
    directory = output_dir or (os.path.dirname(neighbor_pairs_path) or ".")
    prefix = f"{stem}_{name_tag}" if name_tag else stem
    return (
        os.path.join(directory, f"{prefix}_volumes.csv"),
        os.path.join(directory, f"{prefix}_interscellar_volumes.zarr"),
        os.path.join(directory, f"{prefix}_cell_only_volumes.zarr"),
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
    """Write one pair's interscellar and *pairwise* cell-only volumes for visualization.

    The package's global cookie-cutter subtracts every pair's interscellar volume from
    the segmentation at once. That is wrong for inspecting a single pair here, because
    interscellar volumes overlap: cell A would come back pitted with holes carved by its
    other partners, not by this pair.

    So the subtraction is done pairwise -- cell A minus *this* pair's territory in A:

        cell_only_A = A \ T_A(A,B)          cell_only_B = B \ T_B(A,B)

    The corridor never enters either cell, so it plays no part in the subtraction.

    Both volumes are cropped to the pair's own bounding box and share a shape, which is
    all ``visualize_pair_3d`` requires. A one-row ``{stem}_volumes.csv`` is written beside
    them so it can resolve the pair_id to cell IDs without a database.
    """
    import csv as _csv
    import os as _os
    import zarr

    result = compute_interscellar_volume_adaptive(
        mask_3d=mask_3d, cell_a_id=cell_a_id, cell_b_id=cell_b_id,
        pair_id=pair_id, return_debug=True, **kwargs,
    )
    bbox = result['union_bbox']
    crop = mask_3d[bbox]

    interscellar = result['interscellar_mask'].astype(np.uint32) * int(pair_id)
    cell_only = np.zeros(crop.shape, dtype=np.uint32)
    cell_only[(crop == cell_a_id) & ~result['adaptive_a_mask']] = int(cell_a_id)
    cell_only[(crop == cell_b_id) & ~result['adaptive_b_mask']] = int(cell_b_id)

    # Trim to the occupied region so the viewer's bbox scan stays cheap.
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
                        help="Exact per-pair shared/exclusive voxel counts. Retains one packed mask "
                             "per pair in memory; leave off for whole-volume runs.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Worker processes for --pairs mode. Each loads its own copy of the volume.")
    parser.add_argument("--debug", action="store_true", help="Print rho ranges for a single pair")
    args = parser.parse_args(argv)

    if not args.pairs and (args.cell_a is None or args.cell_b is None):
        parser.error("provide --pairs, or both --cell-a and --cell-b")

    voxel_size_um = (args.z, args.y, args.x)

    # Output naming follows compute_interscellar_volumes_3d: every name is derived from
    # the neighbor-pairs file, with --name-tag separating the two pipelines. Explicit
    # --out / --out-zarr still win.
    if args.pairs:
        auto_csv, auto_zarr, _ = default_output_paths(
            args.pairs, args.name_tag, args.output_dir
        )
        if args.out is None:
            args.out = auto_csv
        if args.out_zarr is None and not args.export_pairs:
            args.out_zarr = auto_zarr
        print(f"Output CSV:  {args.out}")
        print(f"Output zarr: {args.out_zarr}")
        for path in (args.out, args.out_zarr):
            if path and "Mobile Documents" in path:
                print(
                    "  WARNING: this path is inside iCloud Drive. Large zarr output there "
                    "will be synced and may stall; pass --output-dir to write locally."
                )
                break

    mask_3d = _load_label_volume(args.mask)
    print(f"Loaded mask {mask_3d.shape} {mask_3d.dtype} from {args.mask}")

    # Metadata built on a different (larger) volume cannot be reindexed onto a crop,
    # so refuse it loudly instead of silently misaligning the surfaces or the bboxes.
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

        for pid in wanted:
            cell_a_id, cell_b_id = lookup[pid]
            out_dir = os.path.join(args.export_dir, f"pair_{pid}")
            info = export_pair_volumes(
                mask_3d, cell_a_id, cell_b_id, pid, out_dir,
                stem=export_stem,
                pad=8,
                voxel_size_um=voxel_size_um,
                global_surface=global_surface,
                halo_bboxes=boxes,
                **geometry,
            )
            print(
                f"pair {pid} (cells {cell_a_id}/{cell_b_id}) -> {out_dir}\n"
                f"  crop {info['shape']} at z,y,x {info['crop_origin_zyx']} | "
                f"interscellar {info['interscellar_voxels']} vox = {info['volume_um3']:.1f} um^3\n"
                f"  cell-only A {info['cell_a_only_voxels']} vox (minus T_A {info['territory_a_voxels']}), "
                f"B {info['cell_b_only_voxels']} vox (minus T_B {info['territory_b_voxels']})"
            )
        print(f"\nDone. Visualize with:\n"
              f"  visualize-pair-3d --pair-id <ID> \\\n"
              f"    --cell-only-zarr {args.export_dir}/pair_<ID>/{export_stem}_cell_only_volumes.zarr \\\n"
              f"    --interscellar-zarr {args.export_dir}/pair_<ID>/{export_stem}_interscellar_volumes.zarr")
        return

    if args.pairs:
        pairs = _load_pairs(args.pairs)
        print(f"Loaded {len(pairs)} pairs from {args.pairs}")

        # Filter once, in the parent: pairs naming a cell absent from this volume cannot
        # be computed at all (the normal case when the pair list covers a larger
        # segmentation than the volume being processed).
        #
        # The bbox table doubles as the list of labels present, which avoids np.unique --
        # that sorts a full copy of the array and is untenable on a whole segmentation.
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

        # Only the boxes for cells actually in a runnable pair need to reach the workers.
        paired_cells = {c for pair in runnable for c in pair[:2]}
        halo_bboxes = {c: b for c, b in halo_bboxes.items() if c in paired_cells}

        worker_kwargs = dict(
            voxel_size_um=voxel_size_um,
            global_surface=global_surface,
            halo_bboxes=halo_bboxes,
            return_debug=bool(args.out_zarr),
            **geometry,
        )

        need_masks = bool(args.out_zarr)
        mask_shape = mask_3d.shape
        volume_gib = mask_3d.nbytes / 2**30

        # Workers slice one pair-local crop at a time, so they can read lazily from the
        # store instead of each holding the whole volume. The parent needed it only for
        # the bbox pass and the face scan, both done by now.
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

        # Resume from a partially written CSV: its pair_ids are the completed work.
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

        store = ds_labels = ds_overlap = None
        written_ids = set()
        if args.out_zarr:
            store, ds_labels, ds_overlap, written_ids = _open_output_zarr(
                args.out_zarr, mask_shape, appending, voxel_size_um, geometry
            )
            if appending and not written_ids:
                print(
                    "  note: --out-zarr has no record of previously written pairs, so pairs "
                    "already listed in the CSV will not be added to it"
                )

        # Chunked streaming: compute a chunk, fold it into the zarr, append its rows,
        # release it. Peak memory is one chunk, not the whole run.
        retained = [] if args.pair_overlap_stats else None
        chunk_size = max(1, args.chunk_size)
        n_done = n_failed = 0
        total_um3 = 0.0
        records = []
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
                    chunk_records, _ = compute_interscellar_volumes_adaptive_for_pairs(
                        mask_3d, chunk, keep_masks=need_masks, verbose=False, **worker_kwargs
                    )
                failures = [r for r in chunk_records if 'error' in r]
                chunk_records = [r for r in chunk_records if 'error' not in r]
                n_failed += len(failures)

                if ds_labels is not None:
                    fresh = []
                    for record in chunk_records:
                        if record.get('interscellar_voxels') and int(record['pair_id']) not in written_ids:
                            _write_pair_into(ds_labels, ds_overlap, record)
                            written_ids.add(int(record['pair_id']))
                            fresh.append(int(record['pair_id']))
                            if retained is not None:
                                retained.append((
                                    record['union_bbox'],
                                    np.packbits(record['interscellar_mask']),
                                    record['interscellar_mask'].shape,
                                    record,
                                ))
                    if fresh:
                        # Persist immediately after the writes so an interrupted run
                        # cannot fold the same pair into overlap_count twice.
                        store.attrs["written_pair_ids"] = sorted(written_ids)

                for record in chunk_records:
                    total_um3 += record.get('total_interscellar_volume_um3', 0.0)
                    record.pop('interscellar_mask', None)
                if csv_writer is not None and retained is None:
                    csv_writer.writerows(chunk_records)
                    csv_handle.flush()
                if retained is None:
                    records.extend({k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                                   for r in chunk_records)

                n_done += len(chunk_records)
                rate = n_done / max(time.time() - started, 1e-6)
                remaining = (len(runnable) - n_done) / rate if rate else 0
                print(
                    f"  {n_done}/{len(runnable)} pairs | {rate:.1f} pairs/s | "
                    f"~{remaining/60:.1f} min left | running total {total_um3:.1f} um^3",
                    flush=True,
                )

            # Exact per-pair overlap needs the final overlap_count, so it happens last.
            if retained is not None:
                print("Computing per-pair overlap against the finished volume...")
                for bbox, packed, shape, record in retained:
                    claim = np.unpackbits(packed, count=int(np.prod(shape))).astype(bool).reshape(shape)
                    counts = np.asarray(ds_overlap[bbox])
                    shared = int((counts[claim] > 1).sum())
                    record['shared_voxels'] = shared
                    record['exclusive_voxels'] = int(record['interscellar_voxels']) - shared
                    records.append({k: v for k, v in record.items() if not isinstance(v, np.ndarray)})
                if csv_writer is not None:
                    csv_writer.writerows(records)
                    csv_handle.flush()
        finally:
            if csv_handle is not None:
                csv_handle.close()

        if n_failed:
            print(f"  {n_failed} pairs failed")
        if args.out:
            print(f"Wrote {n_done} rows to {args.out} (total {total_um3:.2f} um^3)")
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
        print(
            f"Pair {args.cell_a}-{args.cell_b}: {r['interscellar_voxels']} voxels "
            f"({r['interscellar_volume_um3']:.3f} um^3) | "
            f"territory_a={r['territory_a_voxels']} corridor={r['corridor_voxels']} "
            f"territory_b={r['territory_b_voxels']}"
        )
        if args.debug:
            print(f"  rho_a range: {np.nanmin(r['rho_a']):.3f} to {np.nanmax(r['rho_a']):.3f}")
            print(f"  rho_b range: {np.nanmin(r['rho_b']):.3f} to {np.nanmax(r['rho_b']):.3f}")

    # Overlap accounting. Pairs sharing a cell routinely claim the same voxels, so a
    # single-label volume has to arbitrate. Do it before writing the CSV so each pair
    # can report how much of its volume it shares.
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
