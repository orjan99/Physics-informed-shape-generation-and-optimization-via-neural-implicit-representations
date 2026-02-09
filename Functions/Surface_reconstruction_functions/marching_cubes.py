
import numpy as np # type: ignore
import trimesh # type: ignore
from scipy.interpolate import griddata # type: ignore
from skimage import measure # type: ignore
import trimesh.smoothing # type: ignore

def marching_cubes_grid(grid_sdf, grid_points):
   
    """
    Function to perform the marching cubes algorithm and generate a 3D mesh from a implicit surface representation.
    """

    grid_res = int(np.cbrt(len(grid_points))) 
    grid_sdf = grid_sdf.reshape((grid_res, grid_res, grid_res))
    x_coords = grid_points[:, 0]
    y_coords = grid_points[:, 1]
    z_coords = grid_points[:, 2]
    x = np.unique(x_coords) 
    y = np.unique(y_coords)
    z = np.unique(z_coords)

    spacing = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
  
    # Run the marching cubes algorithm to extract the isosurface at SDF == 0.
    verts, faces, normals, values = measure.marching_cubes(grid_sdf, level=0.0, spacing=spacing)

    # Create a trimesh mesh 
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.invert()
    mesh.fill_holes()
    mesh.fix_normals()
    mesh.remove_degenerate_faces()

    #mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, mu=0.5, iterations=10)

    return mesh


def marching_cubes_point_cloud(point_cloud_data,grid_resolution):
    """
    Function to perform the marching cubes algorithm and generate a 3D mesh from a point cloud.
    """
    
    points = point_cloud_data[:, 0:3]
    point_cloud_sdf = point_cloud_data[:, 3]

    tolerance = 1e-9  # Adjust as needed
    inside_mask = point_cloud_sdf <= tolerance

    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    grid_res = grid_resolution

    x = np.linspace(min_bounds[0], max_bounds[0], grid_res)
    y = np.linspace(min_bounds[1], max_bounds[1], grid_res)
    z = np.linspace(min_bounds[2], max_bounds[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  

    grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    grid_sdf = griddata(points, point_cloud_sdf, grid_points, method='linear')
    grid_sdf = np.nan_to_num(grid_sdf, nan=1.0)
    grid_sdf = grid_sdf.reshape((grid_res, grid_res, grid_res))

    spacing = (x[1]-x[0], y[1]-y[0], z[1]-z[0])

    # Run the marching cubes algorithm to extract the isosurface at SDF == 0.
    verts, faces, normals, values = measure.marching_cubes(grid_sdf, level=0.0, spacing=spacing)

    # Create a trimesh mesh 
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fill_holes()
    mesh.fix_normals()
    mesh.remove_degenerate_faces()

    # Optionally smooth the mesh
    mesh = trimesh.smoothing.filter_humphrey(mesh, alpha=0.4, beta=0.5, iterations=20, laplacian_operator=None)

    return mesh