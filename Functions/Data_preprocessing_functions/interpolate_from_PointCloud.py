
from scipy.spatial import cKDTree # type: ignore

def interpolate_from_point_cloud(point_cloud, new_points, quantity = 'SDF'):
    """
    Interpolates data from a point cloud onto a new set of points using nearest neighbor search.
    """

    if quantity == 'SDF':
        point_cloud_quantity = point_cloud[:, 3]     # Extract the SDF values from the point cloud
    elif quantity == 'xdisp':
        point_cloud_quantity = point_cloud[:, 7]
    elif quantity == 'ydisp':
        point_cloud_quantity = point_cloud[:, 8]
    elif quantity == 'zdisp':
        point_cloud_quantity = point_cloud[:, 9]
    elif quantity == 'stress':  
        point_cloud_quantity = point_cloud[:, 10]
    else:
        raise ValueError("Invalid quantity. Choose from 'SDF', 'xdisp', 'ydisp', 'zdisp', or 'stress'.")
        
    
    tree = cKDTree(point_cloud[:, :3])           
    point_cloud_points = point_cloud[:, :3]      
    point_cloud_quantity = point_cloud[:, 3]     

    # Find the nearest neighbors in the point cloud for each point
    distances, indices = tree.query(new_points, k=1) 
    # Get the SDF values for the nearest neighbors
    interpolated_quantities = point_cloud_quantity[indices] 

    return interpolated_quantities