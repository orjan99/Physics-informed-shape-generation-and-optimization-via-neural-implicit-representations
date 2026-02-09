import os
import gc
import numpy as np
import pandas as pd
import trimesh # type: ignore
from scipy.spatial import qhull # type: ignore

from Functions.Data_preprocessing_functions.FEM_interpolation import FEM_to_points_single_load_case_with_SDF_masking
from Functions.Data_preprocessing_functions.PointCloud_generation import generate_point_cloud

# ------- Function to split the point cloud data into chunks for parallel processing - speed up the process -------------------------
def piecewise_interpolation(chunk_pts, chunk_sdf, fem_data, lc):
    return FEM_to_points_single_load_case_with_SDF_masking(chunk_sdf, chunk_pts, fem_data, lc)
#------------------------------------------------------------------------------------------------------------------------


def generate_point_cloud_dataset(num_points, load_case, FEM_data_folder, mesh_data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # 1. Gather all .obj mesh filenames
    all_meshes = [
        f for f in os.listdir(mesh_data_folder)
        if f.lower().endswith('.obj')
    ]

    # 2. Gather all already-processed meshes
    processed = {
        os.path.splitext(f)[0].replace('_PC_data', '')
        for f in os.listdir(output_folder)
        if f.endswith('_PC_data.csv')
    }

    # 3. Build list unprocessed meshes 
    to_do = [m for m in all_meshes if os.path.splitext(m)[0] not in processed]
    skipped = len(all_meshes) - len(to_do)
    total = len(to_do)

    print(f"Skipping {skipped} already-processed meshes; processing {total} new meshes.")

    # Column names for output CSV
    column_names = ['x', 'y', 'z', 'sdf', 'nx', 'ny', 'nz', 'x_disp', 'y_disp', 'z_disp', 'VM_stress']

    error_count = 0
    failed_files = []

    # 4. Loop only over the unprocessed meshes
    for idx, filename in enumerate(to_do, start=1):
        base = os.path.splitext(filename)[0]
        fem_filename = base + "field.csv"
        mesh_input_path = os.path.join(mesh_data_folder, filename)
        fem_input_path  = os.path.join(FEM_data_folder, fem_filename)
        output_path     = os.path.join(output_folder, base + "_PC_data.csv")

        try:
            # -- load mesh --
            mesh = trimesh.load(mesh_input_path)
            mesh.fill_holes()
            mesh.fix_normals()

            # -- load FEM data --
            fem_df = pd.read_csv(fem_input_path)
            fem_np = fem_df.to_numpy()

            # -- generate point cloud --
            pc_data = generate_point_cloud(mesh, num_points)
            pts     = pc_data[:, :3]
            sdf_vals= pc_data[:, 3]

            # -- interpolate FEM onto points --
            fem_pc = FEM_to_points_single_load_case_with_SDF_masking(sdf_vals, pts, fem_np, load_case)

            # -- assemble and save --
            all_data = np.concatenate((pc_data, fem_pc), axis=1)
            pd.DataFrame(all_data, columns=column_names).to_csv(output_path, index=False)

            # Progress print every 10 or last
            if idx % 1 == 0 or idx == total:
                print(f"[{idx}/{total}] Processed {filename}")

        except Exception as e:
            error_count += 1
            failed_files.append(filename)
            print(f"[{idx}/{total}] Error Processing {filename}")

        finally:
            # Clean up large objects safely
            try:
                del mesh, fem_df, fem_np, pc_data, pts, sdf_vals, fem_pc, all_data
            except NameError:
                pass
            gc.collect()


    # Summary
    if error_count:
        print(f"Completed with {error_count} errors. Failed files: {failed_files}")
    else:
        print("All meshes processed successfully.")