import argparse
import os
import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .find_cell_neighbors_3d import (
    create_graph_database,
    export_graph_tables,
    export_to_anndata,
    populate_cells_table,
)


def build_centroid_sphere_neighbor_graph_3d(
    metadata_df: pd.DataFrame,
    radius_um: float,
    voxel_size_um: Tuple[float, float, float] = (0.56, 0.28, 0.28),
    db_path: str = "cell_neighbor_pair_graph_centroid.db",
    cell_id: str = "CellID",
    cell_type: str = "phenotype",
    centroid_x: str = "X_centroid",
    centroid_y: str = "Y_centroid",
    centroid_z: str = "Z_centroid",
) -> sqlite3.Connection:
    required_cols = [cell_id, cell_type, centroid_x, centroid_y, centroid_z]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")

    if radius_um < 0:
        raise ValueError("radius_um must be non-negative")

    conn = create_graph_database(db_path=db_path)
    populate_cells_table(
        conn,
        metadata_df,
        cell_id=cell_id,
        cell_type=cell_type,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        centroid_z=centroid_z,
    )

    cells_df = pd.read_sql_query("SELECT * FROM cells ORDER BY cell_id", conn)
    if len(cells_df) == 0:
        print("No cells found in metadata. Graph contains 0 edges.")
        return conn

    coords_zyx = cells_df[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
    scaled_coords = coords_zyx * np.asarray(voxel_size_um, dtype=float)

    tree = cKDTree(scaled_coords)
    pair_indices = tree.query_pairs(r=radius_um, output_type="set")
    sorted_pairs = sorted(pair_indices)

    print(f"Found {len(sorted_pairs)} centroid-sphere neighbor pairs within {radius_um} um")

    neighbor_rows = []
    for pair_id, (idx_a, idx_b) in enumerate(sorted_pairs, start=1):
        row_a = cells_df.iloc[idx_a]
        row_b = cells_df.iloc[idx_b]
        cell_id_a, cell_id_b = int(row_a["cell_id"]), int(row_b["cell_id"])
        cell_type_a, cell_type_b = row_a["cell_type"], row_b["cell_type"]

        # Keep pair direction stable to match graph edge conventions.
        if cell_id_a > cell_id_b:
            cell_id_a, cell_id_b = cell_id_b, cell_id_a
            cell_type_a, cell_type_b = cell_type_b, cell_type_a

        neighbor_rows.append((pair_id, cell_id_a, cell_id_b, cell_type_a, cell_type_b))

    if neighbor_rows:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO neighbors (pair_id, cell_id_a, cell_id_b, cell_type_a, cell_type_b) VALUES (?, ?, ?, ?, ?)",
            neighbor_rows,
        )
        conn.commit()

    return conn


def run_centroid_neighbor_script(
    metadata_csv_path: str,
    radius_um: float,
    voxel_size_um: Tuple[float, float, float] = (0.56, 0.28, 0.28),
    db_path: Optional[str] = None,
    neighbors_csv: Optional[str] = None,
    cells_csv: Optional[str] = None,
    output_anndata: Optional[str] = None,
    cell_id: str = "CellID",
    cell_type: str = "phenotype",
    centroid_x: str = "X_centroid",
    centroid_y: str = "Y_centroid",
    centroid_z: str = "Z_centroid",
) -> sqlite3.Connection:
    metadata_df = pd.read_csv(metadata_csv_path)

    output_dir = os.path.dirname(metadata_csv_path) or "."
    metadata_stem = os.path.splitext(os.path.basename(metadata_csv_path))[0]
    if db_path is None:
        db_path = os.path.join(output_dir, f"{metadata_stem}_centroid_neighbor_graph_3d.db")
    if neighbors_csv is None:
        neighbors_csv = os.path.join(output_dir, f"{metadata_stem}_centroid_neighbors_3d.csv")

    print(f"Metadata: {metadata_csv_path}")
    print(f"Radius: {radius_um} um")
    print(f"Voxel size (z,y,x): {voxel_size_um}")
    print(f"Database output: {db_path}")

    conn = create_neighbor_edge_table_database_centroid_3d(
        metadata_df=metadata_df,
        radius_um=radius_um,
        voxel_size_um=voxel_size_um,
        db_path=db_path,
        output_csv=neighbors_csv,
        output_anndata=output_anndata,
        output_cells_csv=cells_csv,
        cell_id=cell_id,
        cell_type=cell_type,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        centroid_z=centroid_z,
    )

    return conn


def create_neighbor_edge_table_database_centroid_3d(
    metadata_df: pd.DataFrame,
    radius_um: float,
    voxel_size_um: Tuple[float, float, float] = (0.56, 0.28, 0.28),
    db_path: str = "cell_neighbor_pair_graph_centroid.db",
    output_csv: Optional[str] = None,
    output_anndata: Optional[str] = None,
    output_cells_csv: Optional[str] = None,
    cell_id: str = "CellID",
    cell_type: str = "phenotype",
    centroid_x: str = "X_centroid",
    centroid_y: str = "Y_centroid",
    centroid_z: str = "Z_centroid",
) -> sqlite3.Connection:
    conn = build_centroid_sphere_neighbor_graph_3d(
        metadata_df=metadata_df,
        radius_um=radius_um,
        voxel_size_um=voxel_size_um,
        db_path=db_path,
        cell_id=cell_id,
        cell_type=cell_type,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        centroid_z=centroid_z,
    )

    if output_csv:
        if output_cells_csv:
            export_graph_tables(conn, cells_file=output_cells_csv, neighbors_file=output_csv)
            print(f"Neighbors CSV: {output_csv}")
            print(f"Cells CSV: {output_cells_csv}")
        else:
            df_neighbors = pd.read_sql_query("SELECT * FROM neighbors ORDER BY pair_id", conn)
            df_neighbors.to_csv(output_csv, index=False)
            print(f"Neighbors CSV: {output_csv}")

    if output_anndata:
        export_to_anndata(conn, output_file=output_anndata)

    return conn


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 3D cell-neighbor graph by centroid sphere search. "
            "Outputs use the same node/edge graph format as the surface-based pipeline."
        )
    )
    parser.add_argument("--metadata-csv", required=True, help="Path to metadata CSV")
    parser.add_argument(
        "--radius-um",
        required=True,
        type=float,
        help="Centroid neighbor search radius in micrometers",
    )
    parser.add_argument(
        "--voxel-size-um",
        nargs=3,
        type=float,
        default=(0.56, 0.28, 0.28),
        metavar=("Z", "Y", "X"),
        help="Voxel size in micrometers as three values: z y x",
    )
    parser.add_argument("--db-path", default=None, help="Output SQLite graph database path")
    parser.add_argument("--neighbors-csv", default=None, help="Output neighbors edge CSV path")
    parser.add_argument(
        "--cells-csv",
        default=None,
        help="Optional output cells node CSV path (only written when provided)",
    )
    parser.add_argument("--output-anndata", default=None, help="Optional output .h5ad file path")

    parser.add_argument("--cell-id-col", default="CellID", help="Metadata cell id column")
    parser.add_argument("--cell-type-col", default="phenotype", help="Metadata cell type column")
    parser.add_argument("--centroid-x-col", default="X_centroid", help="Metadata centroid x column")
    parser.add_argument("--centroid-y-col", default="Y_centroid", help="Metadata centroid y column")
    parser.add_argument("--centroid-z-col", default="Z_centroid", help="Metadata centroid z column")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    conn = run_centroid_neighbor_script(
        metadata_csv_path=args.metadata_csv,
        radius_um=args.radius_um,
        voxel_size_um=tuple(args.voxel_size_um),
        db_path=args.db_path,
        neighbors_csv=args.neighbors_csv,
        cells_csv=args.cells_csv,
        output_anndata=args.output_anndata,
        cell_id=args.cell_id_col,
        cell_type=args.cell_type_col,
        centroid_x=args.centroid_x_col,
        centroid_y=args.centroid_y_col,
        centroid_z=args.centroid_z_col,
    )
    conn.close()
    print("Centroid neighbor graph build complete.")


if __name__ == "__main__":
    main()
