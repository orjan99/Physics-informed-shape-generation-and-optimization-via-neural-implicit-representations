import numpy as np

def strategic_sampling(points, SDF,
                       percentage_inside,
                       percentage_near_surface,
                       percentage_outisde):
    """
    Function to strategically sample points for point cloud generation from the domain to get an even distribution
    of points inside, near the surface, and outside the surface.

    The function ensures that ALL POINTS ON THE INSIDE OF THE SURFACE ARE INCLUDED, 
    and that this amount of points is equal to the specified percentage of the total number of points.
    """

    # Define the masks for each region --> What is considered inside, near surface, and outside
    mask_inside      = SDF < 1e-6
    mask_near_surface= (SDF >= 0) & (SDF <= 1)
    mask_outisde     = SDF > 1 

    # Convert the percanteges to fractions
    percentage_inside      = percentage_inside      / 100
    percentage_near_surface= percentage_near_surface/ 100
    percentage_outisde     = percentage_outisde     / 100

    # Get the indices in each region
    idx_inside       = np.where(mask_inside)[0]
    idx_near_surface = np.where(mask_near_surface)[0]
    idx_outisde      = np.where(mask_outisde)[0]

    # Get the number of points in each region based on the percentages given by the user
    num_inside    = len(idx_inside)  ## Include all the points inside the surface 
    num_near_surface = int(num_inside * (percentage_near_surface / percentage_inside))
    num_outisde      = int(num_inside * (percentage_outisde    / percentage_inside))

    # Sample the indices in each region
    chosen_inside       = idx_inside
    chosen_near_surface = np.random.choice(idx_near_surface,size=min(num_near_surface, len(idx_near_surface)),replace=False)
    chosen_outisde      = np.random.choice(idx_outisde,size=min(num_outisde, len(idx_outisde)),replace=False)

    # Combine sampled indices
    sampled_idx = np.concatenate((chosen_inside,chosen_near_surface,chosen_outisde))

    # Shuffle the indices to get a random distribution
    np.random.shuffle(sampled_idx)

    # Slice both the points and SDF arrays
    sampled_points = points[sampled_idx]
    sampled_SDF    = SDF[sampled_idx]

    # Calculate the percentage of points in each region
    total_points = len(sampled_idx)
    percentage_inside_true      = len(chosen_inside)       / total_points
    percentage_near_surface_true= len(chosen_near_surface) / total_points
    percentage_outisde_true     = len(chosen_outisde)      / total_points

    print(f"Number of Points inside: {len(chosen_inside)}, Percentage: {percentage_inside_true:.2f}")
    print(f"Number of Points near surface: {len(chosen_near_surface)}, Percentage: {percentage_near_surface_true:.2f}")
    print(f"Number of Points outside: {len(chosen_outisde)}, Percentage: {percentage_outisde_true:.2f}")
    print(f"Total number of points: {total_points}")
    print(f"Reduced the number of points from {len(points)} to {total_points} points") 

    return sampled_points, sampled_SDF