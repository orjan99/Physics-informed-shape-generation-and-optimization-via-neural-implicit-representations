
import numpy as np
import plotly.graph_objects as go # type: ignore

def generate_grid(mesh, grid_resolution):

    # Determine the bounding box of grid points and extend the bounding box by adding a 10% margin
    max_bounds = mesh.bounds[1]
    min_bounds = mesh.bounds[0]

    # Extend the bounding box by 5% in each direction
    max_bounds = max_bounds + 0.05 * (max_bounds - min_bounds)
    min_bounds = min_bounds - 0.05 * (max_bounds - min_bounds) 

    grid_res = grid_resolution #  = Number of points in each dimension

    # Create a regular grid over the bounding box
    x = np.linspace(min_bounds[0], max_bounds[0], grid_res)
    y = np.linspace(min_bounds[1], max_bounds[1], grid_res)
    z = np.linspace(min_bounds[2], max_bounds[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # using 'ij' ordering

    # combine the grid points into a single numpy array (x,y,z)
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return grid_points


