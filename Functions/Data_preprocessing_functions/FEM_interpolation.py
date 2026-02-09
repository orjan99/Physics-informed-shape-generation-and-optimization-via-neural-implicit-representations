
import numpy as np
import trimesh # type: ignore
from scipy.interpolate import LinearNDInterpolator # type: ignore



# Function 1 --> FEM to grid points - WITH SDF MASKING - Single Load Case: ---------------------------------------------------------------
def FEM_to_points_single_load_case_with_SDF_masking(sdf,point_coordinates,FEM_data, load_case):
    """
    Interpolates the FEM data to the grid points for a single load case.

    Parameters:
    sdf  = SDF values at the grid points
    point_coordinates = coordinates of the grid points
    FEM_data = FEM data from the simulation
    load_case = string indicating the load case ('horizontal', 'vertical', 'diagonal', 'torsional')

    Returns:
    grid_FEM = interpolated FEM data at the grid points for the specified load case
    """
    
    # Select the values based on the load case  
    if load_case == 'horizontal':
        x_disp = FEM_data[:, 10]
        y_disp = FEM_data[:, 11]
        z_disp = FEM_data[:, 12]
        stress = FEM_data[:, 14]
    elif load_case == 'vertical':
        x_disp = FEM_data[:,  5]
        y_disp = FEM_data[:,  6]
        z_disp = FEM_data[:,  7]
        stress = FEM_data[:,  9]
    elif load_case == 'diagonal':
        x_disp = FEM_data[:, 15]
        y_disp = FEM_data[:, 16]
        z_disp = FEM_data[:, 17]
        stress = FEM_data[:, 19]
    elif load_case == 'torsional':
        x_disp = FEM_data[:, 20]
        y_disp = FEM_data[:, 21]
        z_disp = FEM_data[:, 22]
        stress = FEM_data[:, 24]
    else:
        raise ValueError(
            "Invalid load case. "
            "Choose from 'horizontal', 'vertical', 'diagonal', or 'torsional'."
        )
 
    points = point_coordinates

    #1. Extract the x,y,z coordinates from the FEM data
    x_values = FEM_data[:,2]
    y_values = FEM_data[:,3]
    z_values = FEM_data[:,4]
    xyz_FEM = np.column_stack((x_values, y_values, z_values)) # Store FEM nodes

    # Create an interpolator for each component of the displacement and stress
    interp_hor_xdisp = LinearNDInterpolator(xyz_FEM,  x_disp)
    interp_hor_ydisp = LinearNDInterpolator(xyz_FEM,  y_disp)
    interp_hor_zdisp = LinearNDInterpolator(xyz_FEM,  z_disp) 
    interp_hor_stress = LinearNDInterpolator(xyz_FEM, stress)

    #2. Interpolate the values at the grid points
    hor_xdisp_new = interp_hor_xdisp(points)
    hor_ydisp_new = interp_hor_ydisp(points)
    hor_zdisp_new = interp_hor_zdisp(points)
    hor_stress_new = interp_hor_stress(points)

    #3. Handle NaN values by replacing them with 0
    hor_xdisp_new = np.nan_to_num(hor_xdisp_new)
    hor_ydisp_new = np.nan_to_num(hor_ydisp_new)
    hor_zdisp_new = np.nan_to_num(hor_zdisp_new)
    hor_stress_new = np.nan_to_num(hor_stress_new)


    # Define a tolerance for being "on the surface"
    tolerance = 1e-9  
    inside_mask = sdf <= tolerance

    # Set the values for points outside the mesh to zero
    hor_xdisp_new[~inside_mask] = 0
    hor_ydisp_new[~inside_mask] = 0
    hor_zdisp_new[~inside_mask] = 0
    hor_stress_new[~inside_mask] = 0

    # Save the interpolated data to a new array 
    grid_FEM = np.column_stack((hor_xdisp_new, hor_ydisp_new, hor_zdisp_new,hor_stress_new))

    return grid_FEM

# -------------------------------------------------------------------------------------------------------------------------------------



    
# Function 2 --> FEM to grid points - WITHOUT SDF Mask - Single Case ----------------------------------------------------------------
def FEM_to_points_single_case_without_SDF_masking(sdf,point_coordinates,FEM_data, load_case):
    """
    Interpolates the FEM data to the grid points for a single load case.

    Parameters:
    sdf  = SDF values at the grid points
    point_coordinates = coordinates of the grid points
    FEM_data = FEM data from the simulation
    load_case = string indicating the load case ('horizontal', 'vertical', 'diagonal', 'torsional')

    Returns:
    grid_FEM = interpolated FEM data at the grid points for the specified load case
    """
    
    # Select the values based on the load case  
    if load_case == 'horizontal':
        x_disp = FEM_data[:, 10]
        y_disp = FEM_data[:, 11]
        z_disp = FEM_data[:, 12]
        stress = FEM_data[:, 14]
    elif load_case == 'vertical':
        x_disp = FEM_data[:,  5]
        y_disp = FEM_data[:,  6]
        z_disp = FEM_data[:,  7]
        stress = FEM_data[:,  9]
    elif load_case == 'diagonal':
        x_disp = FEM_data[:, 15]
        y_disp = FEM_data[:, 16]
        z_disp = FEM_data[:, 17]
        stress = FEM_data[:, 19]
    elif load_case == 'torsional':
        x_disp = FEM_data[:, 20]
        y_disp = FEM_data[:, 21]
        z_disp = FEM_data[:, 22]
        stress = FEM_data[:, 24]
    else:
        raise ValueError(
            "Invalid load case. "
            "Choose from 'horizontal', 'vertical', 'diagonal', or 'torsional'."
        )
 
    points = point_coordinates

    #1. Extract the x,y,z coordinates from the FEM data
    x_values = FEM_data[:,2]
    y_values = FEM_data[:,3]
    z_values = FEM_data[:,4]
    xyz_FEM = np.column_stack((x_values, y_values, z_values)) # Store FEM nodes

    # Create an interpolator for each component of the displacement and stress
    interp_hor_xdisp = LinearNDInterpolator(xyz_FEM,  x_disp)
    interp_hor_ydisp = LinearNDInterpolator(xyz_FEM,  y_disp)
    interp_hor_zdisp = LinearNDInterpolator(xyz_FEM,  z_disp) 
    interp_hor_stress = LinearNDInterpolator(xyz_FEM, stress)

    #2. Interpolate the values at the grid points
    hor_xdisp_new = interp_hor_xdisp(points)
    hor_ydisp_new = interp_hor_ydisp(points)
    hor_zdisp_new = interp_hor_zdisp(points)
    hor_stress_new = interp_hor_stress(points)

    #3. Handle NaN values by replacing them with 0
    hor_xdisp_new = np.nan_to_num(hor_xdisp_new)
    hor_ydisp_new = np.nan_to_num(hor_ydisp_new)
    hor_zdisp_new = np.nan_to_num(hor_zdisp_new)
    hor_stress_new = np.nan_to_num(hor_stress_new)

    # Save the interpolated data to a new array 
    grid_FEM = np.column_stack((hor_xdisp_new, hor_ydisp_new, hor_zdisp_new,hor_stress_new))

    return grid_FEM
# -------------------------------------------------------------------------------------------------------------------------------------


# Function 3 --> FEM to grid points - All load cases - SDF Masking
def FEM_to_grid_all_load_cases_with_SDF_masking(sdf,point_coordinates,FEM_data):
    """
    Interpolates the FEM data to the grid points for all load cases.

    Parameters:
    sdf  = SDF values at the grid points
    point_coordinates = coordinates of the grid points
    FEM_data = FEM data from the simulation

    Returns:
    grid_FEM = interpolated FEM data at the grid points for all load cases
    """
    
    points = point_coordinates

    #1. Extract the x,y,z coordinates from the FEM data
    x_values = FEM_data[:,2]
    y_values = FEM_data[:,3]
    z_values = FEM_data[:,4]
    xyz_FEM = np.column_stack((x_values, y_values, z_values)) # Store FEM nodes

    # Create interpolators for each of the results in the FEM data
    interp_ver_xdisp = LinearNDInterpolator(  xyz_FEM, FEM_data[:,5])
    interp_ver_ydisp = LinearNDInterpolator(  xyz_FEM, FEM_data[:,6])
    interp_ver_zdisp = LinearNDInterpolator(  xyz_FEM, FEM_data[:,7])
    interp_ver_magdisp = LinearNDInterpolator(  xyz_FEM, FEM_data[:,8])
    interp_ver_stress = LinearNDInterpolator( xyz_FEM, FEM_data[:,9])  

    interp_hor_xdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,10])
    interp_hor_ydisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,11])
    interp_hor_zdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,12]) 
    interp_hor_stress = LinearNDInterpolator( xyz_FEM, FEM_data[:,14])
    interp_hor_magdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,13])

    interp_dia_xdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,15])
    interp_dia_ydisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,16]) 
    interp_dia_zdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,17]) 
    interp_dia_magdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,18])
    interp_dia_stress = LinearNDInterpolator( xyz_FEM, FEM_data[:,19])

    interp_tor_xdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,20])
    interp_tor_ydisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,21]) 
    interp_tor_zdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,22]) 
    interp_tor_magdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,23])
    interp_tor_stress = LinearNDInterpolator( xyz_FEM, FEM_data[:,24])

    # Interpolate displacements at the point cloud coordinates
    ver_xdisp_new = interp_ver_xdisp(points)
    ver_ydisp_new = interp_ver_ydisp(points)
    ver_zdisp_new = interp_ver_zdisp(points)
    ver_stress_new = interp_ver_stress(points)
    ver_magdisp_new = interp_ver_magdisp(points)

    hor_xdisp_new = interp_hor_xdisp(points)
    hor_ydisp_new = interp_hor_ydisp(points)
    hor_zdisp_new = interp_hor_zdisp(points)
    hor_stress_new = interp_hor_stress(points)
    hor_magdisp_new = interp_hor_magdisp(points)

    dia_xdisp_new = interp_dia_xdisp(points)
    dia_ydisp_new = interp_dia_ydisp(points)
    dia_zdisp_new = interp_dia_zdisp(points)
    dia_stress_new = interp_dia_stress(points)
    dia_magdisp_new = interp_dia_magdisp(points)

    tor_xdisp_new = interp_tor_xdisp(points)
    tor_ydisp_new = interp_tor_ydisp(points)
    tor_zdisp_new = interp_tor_zdisp(points)
    tor_stress_new = interp_tor_stress(points)
    tor_magdisp_new = interp_tor_magdisp(points)

    # Points outside the mesh will have NaN displacements and stresses
    # Replace NaN values with zeros
    ver_xdisp_new = np.nan_to_num(ver_xdisp_new)
    ver_ydisp_new = np.nan_to_num(ver_ydisp_new)
    ver_zdisp_new = np.nan_to_num(ver_zdisp_new)
    ver_stress_new = np.nan_to_num(ver_stress_new)
    ver_magdisp_new = np.nan_to_num(ver_magdisp_new)

    hor_xdisp_new = np.nan_to_num(hor_xdisp_new)
    hor_ydisp_new = np.nan_to_num(hor_ydisp_new)
    hor_zdisp_new = np.nan_to_num(hor_zdisp_new)
    hor_stress_new = np.nan_to_num(hor_stress_new)
    hor_magdisp_new = np.nan_to_num(hor_magdisp_new)

    dia_xdisp_new = np.nan_to_num(dia_xdisp_new)
    dia_ydisp_new = np.nan_to_num(dia_ydisp_new)
    dia_zdisp_new = np.nan_to_num(dia_zdisp_new)
    dia_stress_new = np.nan_to_num(dia_stress_new)
    dia_magdisp_new = np.nan_to_num(dia_magdisp_new)

    tor_xdisp_new = np.nan_to_num(tor_xdisp_new)
    tor_ydisp_new = np.nan_to_num(tor_ydisp_new)
    tor_zdisp_new = np.nan_to_num(tor_zdisp_new)
    tor_stress_new = np.nan_to_num(tor_stress_new)
    tor_magdisp_new = np.nan_to_num(tor_magdisp_new)

    tolerance = 1e-9  
    inside_mask = sdf <= tolerance

    ver_xdisp_new[~inside_mask] = 0
    ver_ydisp_new[~inside_mask] = 0
    ver_zdisp_new[~inside_mask] = 0
    ver_stress_new[~inside_mask] = 0
    ver_magdisp_new[~inside_mask] = 0

    hor_xdisp_new[~inside_mask] = 0
    hor_ydisp_new[~inside_mask] = 0
    hor_zdisp_new[~inside_mask] = 0
    hor_stress_new[~inside_mask] = 0
    hor_magdisp_new[~inside_mask] = 0

    dia_xdisp_new[~inside_mask] = 0
    dia_ydisp_new[~inside_mask] = 0
    dia_zdisp_new[~inside_mask] = 0
    dia_stress_new[~inside_mask] = 0
    dia_magdisp_new[~inside_mask] = 0

    tor_xdisp_new[~inside_mask] = 0
    tor_ydisp_new[~inside_mask] = 0
    tor_zdisp_new[~inside_mask] = 0
    tor_stress_new[~inside_mask] = 0
    tor_magdisp_new[~inside_mask] = 0

    #4. Store the interpolated values in a np.array
    grid_FEM = np.column_stack((ver_xdisp_new, ver_ydisp_new, ver_zdisp_new, ver_stress_new, ver_magdisp_new,
                            hor_xdisp_new, hor_ydisp_new, hor_zdisp_new, hor_stress_new, hor_magdisp_new,
                            dia_xdisp_new, dia_ydisp_new, dia_zdisp_new, dia_stress_new, dia_magdisp_new,
                            tor_xdisp_new, tor_ydisp_new, tor_zdisp_new, tor_stress_new, tor_magdisp_new))
    return grid_FEM
# -------------------------------------------------------------------------------------------------------------------------------------

# Function 4 --> FEM to grid points - WITHOUT SDF Mask - All load cases

def FEM_to_grid_all_load_cases_without_SDF_masking(sdf,point_coordinates,FEM_data):
    """
    Interpolates the FEM data to the grid points for all load cases.

    Parameters:
    sdf  = SDF values at the grid points
    point_coordinates = coordinates of the grid points
    FEM_data = FEM data from the simulation

    Returns:
    grid_FEM = interpolated FEM data at the grid points for all load cases
    """
    
    points = point_coordinates

    #1. Extract the x,y,z coordinates from the FEM data
    x_values = FEM_data[:,2]
    y_values = FEM_data[:,3]
    z_values = FEM_data[:,4]
    xyz_FEM = np.column_stack((x_values, y_values, z_values)) # Store FEM nodes

    # Create interpolators for each of the results in the FEM data
    interp_ver_xdisp = LinearNDInterpolator(  xyz_FEM, FEM_data[:,5])
    interp_ver_ydisp = LinearNDInterpolator(  xyz_FEM, FEM_data[:,6])
    interp_ver_zdisp = LinearNDInterpolator(  xyz_FEM, FEM_data[:,7])
    interp_ver_magdisp = LinearNDInterpolator(  xyz_FEM, FEM_data[:,8])
    interp_ver_stress = LinearNDInterpolator( xyz_FEM, FEM_data[:,9])  

    interp_hor_xdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,10])
    interp_hor_ydisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,11])
    interp_hor_zdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,12]) 
    interp_hor_stress = LinearNDInterpolator( xyz_FEM, FEM_data[:,14])
    interp_hor_magdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,13])

    interp_dia_xdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,15])
    interp_dia_ydisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,16]) 
    interp_dia_zdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,17]) 
    interp_dia_magdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,18])
    interp_dia_stress = LinearNDInterpolator( xyz_FEM, FEM_data[:,19])

    interp_tor_xdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,20])
    interp_tor_ydisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,21]) 
    interp_tor_zdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,22]) 
    interp_tor_magdisp = LinearNDInterpolator( xyz_FEM, FEM_data[:,23])
    interp_tor_stress = LinearNDInterpolator( xyz_FEM, FEM_data[:,24])

    # Interpolate displacements at the point cloud coordinates
    ver_xdisp_new = interp_ver_xdisp(points)
    ver_ydisp_new = interp_ver_ydisp(points)
    ver_zdisp_new = interp_ver_zdisp(points)
    ver_stress_new = interp_ver_stress(points)
    ver_magdisp_new = interp_ver_magdisp(points)

    hor_xdisp_new = interp_hor_xdisp(points)
    hor_ydisp_new = interp_hor_ydisp(points)
    hor_zdisp_new = interp_hor_zdisp(points)
    hor_stress_new = interp_hor_stress(points)
    hor_magdisp_new = interp_hor_magdisp(points)

    dia_xdisp_new = interp_dia_xdisp(points)
    dia_ydisp_new = interp_dia_ydisp(points)
    dia_zdisp_new = interp_dia_zdisp(points)
    dia_stress_new = interp_dia_stress(points)
    dia_magdisp_new = interp_dia_magdisp(points)

    tor_xdisp_new = interp_tor_xdisp(points)
    tor_ydisp_new = interp_tor_ydisp(points)
    tor_zdisp_new = interp_tor_zdisp(points)
    tor_stress_new = interp_tor_stress(points)
    tor_magdisp_new = interp_tor_magdisp(points)

    # Points outside the mesh will have NaN displacements and stresses
    # Replace NaN values with zeros
    ver_xdisp_new = np.nan_to_num(ver_xdisp_new)
    ver_ydisp_new = np.nan_to_num(ver_ydisp_new)
    ver_zdisp_new = np.nan_to_num(ver_zdisp_new)
    ver_stress_new = np.nan_to_num(ver_stress_new)
    ver_magdisp_new = np.nan_to_num(ver_magdisp_new)

    hor_xdisp_new = np.nan_to_num(hor_xdisp_new)
    hor_ydisp_new = np.nan_to_num(hor_ydisp_new)
    hor_zdisp_new = np.nan_to_num(hor_zdisp_new)
    hor_stress_new = np.nan_to_num(hor_stress_new)
    hor_magdisp_new = np.nan_to_num(hor_magdisp_new)

    dia_xdisp_new = np.nan_to_num(dia_xdisp_new)
    dia_ydisp_new = np.nan_to_num(dia_ydisp_new)
    dia_zdisp_new = np.nan_to_num(dia_zdisp_new)
    dia_stress_new = np.nan_to_num(dia_stress_new)
    dia_magdisp_new = np.nan_to_num(dia_magdisp_new)

    tor_xdisp_new = np.nan_to_num(tor_xdisp_new)
    tor_ydisp_new = np.nan_to_num(tor_ydisp_new)
    tor_zdisp_new = np.nan_to_num(tor_zdisp_new)
    tor_stress_new = np.nan_to_num(tor_stress_new)
    tor_magdisp_new = np.nan_to_num(tor_magdisp_new)

    #4. Store the interpolated values in a np.array
    grid_FEM = np.column_stack((ver_xdisp_new, ver_ydisp_new, ver_zdisp_new, ver_stress_new, ver_magdisp_new,
                            hor_xdisp_new, hor_ydisp_new, hor_zdisp_new, hor_stress_new, hor_magdisp_new,
                            dia_xdisp_new, dia_ydisp_new, dia_zdisp_new, dia_stress_new, dia_magdisp_new,
                            tor_xdisp_new, tor_ydisp_new, tor_zdisp_new, tor_stress_new, tor_magdisp_new))
    return grid_FEM
  


    



