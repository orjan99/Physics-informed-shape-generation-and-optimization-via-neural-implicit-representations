from scipy.interpolate import griddata # type: ignore 
import torch
import numpy as np
from skimage import measure # type: ignore 
import trimesh # type: ignore 
import os
import pandas as pd
from File_Paths.file_paths import point_cloud_path 
from scipy.interpolate import griddata # type: ignore 

def generate_mesh_from_GINN(model, test_case, 
                            grid_resolution, 
                            device,
                            smoothing = False,
                            validation_mode=False,
                            Validation_point_cloud_filename=None):  
    """
    Generate a mesh using Marching Cubes from a SDF model or interpolated SDF.
    """
    # Load and scale point cloud if validating
    if validation_mode == True:
        if Validation_point_cloud_filename is None:
            raise ValueError("Validation_point_cloud_filename must be provided in validation mode.")
        domain_scaling = test_case.domain_scaling_factor
        domain_center = test_case.domain_center

        file_path = os.path.join(point_cloud_path, Validation_point_cloud_filename) 
        point_cloud = pd.read_csv(file_path).to_numpy()

        point_cloud[:, :3] = (point_cloud[:, :3] - domain_center) * domain_scaling
        point_cloud[:, 3] *= domain_scaling
        x_min, y_min, z_min = point_cloud[:, :3].min(axis=0)
        x_max, y_max, z_max = point_cloud[:, :3].max(axis=0)

        sdf = griddata(point_cloud[:, :3], point_cloud[:, 3], coords_flat, method='linear')
        sdf = np.nan_to_num(sdf, nan=1.0)
        SDF_grid = sdf.reshape(nx, ny, nz)
    else:
        if test_case.Symmetry == False: 
            domain = test_case.domain
            x_min, x_max = domain[0], domain[1]
            y_min, y_max = domain[2], domain[3]
            z_min, z_max = domain[4], domain[5] 

            # Compute the number of points on each axis based on the grid resolution and the bounding box 
            dx = x_max - x_min
            dy = y_max - y_min
            dz = z_max - z_min
            max_dim = max(dx, dy, dz)
            scale = grid_resolution / max_dim

            nx = int(np.ceil(dx * scale))
            ny = int(np.ceil(dy * scale))
            nz = int(np.ceil(dz * scale)) 

            x = np.linspace(x_min, x_max, nx)
            y = np.linspace(y_min, y_max, ny)
            z = np.linspace(z_min, z_max, nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            coords_flat = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

            coords = torch.tensor(coords_flat, dtype=torch.float32, device=device)
            model.eval()   

            with torch.no_grad():
                sdf = model(coords)
                SDF_grid = sdf.cpu().numpy().reshape(nx, ny, nz)
                spacing = ((x[1] - x[0]), (y[1] - y[0]), (z[1] - z[0])) 
        else:
            x_min_full = -test_case.domain[1] if 'x' in test_case.symmetry_axis else test_case.domain[0]
            x_max_full =  test_case.domain[1]
            y_min_full = -test_case.domain[3] if 'y' in test_case.symmetry_axis else test_case.domain[2]
            y_max_full =  test_case.domain[3]
            z_min_full = -test_case.domain[5] if 'z' in test_case.symmetry_axis else test_case.domain[4]
            z_max_full =  test_case.domain[5] 

            dx_full = x_max_full - x_min_full
            dy_full = y_max_full - y_min_full
            dz_full = z_max_full - z_min_full 
            max_full = max(dx_full, dy_full, dz_full)
            scale = grid_resolution / max_full 

            nx_full = int(np.ceil((x_max_full - x_min_full) * scale))
            ny_full = int(np.ceil((y_max_full - y_min_full) * scale))
            nz_full = int(np.ceil((z_max_full - z_min_full) * scale))

            x_full = np.linspace(x_min_full, x_max_full, nx_full)
            y_full = np.linspace(y_min_full, y_max_full, ny_full)
            z_full = np.linspace(z_min_full, z_max_full, nz_full)

            spacing = (
                x_full[1] - x_full[0],
                y_full[1] - y_full[0],
                z_full[1] - z_full[0]
            ) 

            # 2) Build reduced‐domain grid 
            x_min, x_max = test_case.domain[0], test_case.domain[1]
            y_min, y_max = test_case.domain[2], test_case.domain[3]
            z_min, z_max = test_case.domain[4], test_case.domain[5] 

            dx = x_max - x_min
            dy = y_max - y_min
            dz = z_max - z_min 

            nx = int(np.ceil(dx * scale))
            ny = int(np.ceil(dy * scale))
            nz = int(np.ceil(dz * scale)) 

            x = np.linspace(x_min, x_max, nx)
            y = np.linspace(y_min, y_max, ny)
            z = np.linspace(z_min, z_max, nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            coords_flat = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]) 
            coords_flat = torch.tensor(coords_flat, dtype=torch.float32,device=device) 

            model.eval()
            with torch.no_grad():
                sdf_reduced, points_full, sdf_full = model(coords_flat)  
            
            pts = points_full.cpu().numpy()    # (M,3)
            vals = sdf_full.cpu().numpy().ravel()  # (M,)


            ix = np.round((pts[:,0] - x_min_full) / spacing[0]).astype(int)
            iy = np.round((pts[:,1] - y_min_full) / spacing[1]).astype(int)
            iz = np.round((pts[:,2] - z_min_full) / spacing[2]).astype(int)

            SDF_grid = np.empty((nx_full, ny_full, nz_full), dtype=vals.dtype)
            SDF_grid[ix, iy, iz] = vals 
            
            

                
    # Marching Cubes algorithm to extract the mesh from the SDF grid 

    verts, faces, normals, values = measure.marching_cubes(SDF_grid, level=0.0, spacing=spacing)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.invert()
    mesh.fill_holes()
    mesh.fix_normals()

    if smoothing == True: 
        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=10) 
    
    return mesh 