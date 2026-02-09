
import pandas as pd
from plotly.subplots import make_subplots # type: ignore
import plotly.graph_objects as go # type: ignore
import numpy as np

# Function 1 --> Plot FEM data -> Only the filtered point cloud (SDF <= 0) is plotted

def plot_fem_data(FEM_data, point_coordinates , sdf ,grid_FEM, plot_quantity,load_case):

    # Select the values based on the load case  
    if load_case == 'horizontal':
        x_disp = FEM_data[:, 10]
        y_disp = FEM_data[:, 11]
        z_disp = FEM_data[:, 12]
        stress = FEM_data[:, 14]
        original_FEM = np.column_stack((x_disp, y_disp, z_disp, stress))
    elif load_case == 'vertical':
        x_disp = FEM_data[:,  5]
        y_disp = FEM_data[:,  6]
        z_disp = FEM_data[:,  7]
        stress = FEM_data[:,  9]
        original_FEM = np.column_stack((x_disp, y_disp, z_disp, stress))
    elif load_case == 'diagonal':
        x_disp = FEM_data[:, 15]
        y_disp = FEM_data[:, 16]
        z_disp = FEM_data[:, 17]
        stress = FEM_data[:, 19]
        original_FEM = np.column_stack((x_disp, y_disp, z_disp, stress))
    elif load_case == 'torsional':
        x_disp = FEM_data[:, 20]
        y_disp = FEM_data[:, 21]
        z_disp = FEM_data[:, 22]
        stress = FEM_data[:, 24]
        original_FEM = np.column_stack((x_disp, y_disp, z_disp, stress))
    else:
        raise ValueError(
            "Invalid load case. "
            "Choose from 'horizontal', 'vertical', 'diagonal', or 'torsional'."
        )


    # format plot based on the selected quantity
    if plot_quantity == 'stress':
        title = 'Von-Mises Stress'
        sub_plot_titles = ('Interpolated Von-Mises Stress', 'Original Von-Mises Stress')
        color_quantity = grid_FEM[:,3]  
        global_min = min(grid_FEM[:,3].min(),original_FEM[:,3].min())
        global_max = max(grid_FEM[:,3].max(),original_FEM[:,3].max())
        index = 3
        

    elif plot_quantity == 'x displacement':
        title = 'X Displacement'
        sub_plot_titles = ('Interpolated X Displacement', 'Original X Displacement')
        color_quantity = grid_FEM[:,0]  
        global_min = min(grid_FEM[:,0].min(),original_FEM[:,0].min())
        global_max = max(grid_FEM[:,0].max(),original_FEM[:,0].max())
        index = 0

    elif plot_quantity == 'y displacement':
        title = 'Y Displacement'
        sub_plot_titles = ('Interpolated Y Displacement', 'Original Y Displacement')
        color_quantity = grid_FEM[:,1]  
        global_min = min(grid_FEM[:,1].min(),original_FEM[:,1].min())
        global_max = max(grid_FEM[:,1].max(),original_FEM[:,1].max())
        index = 1

    elif plot_quantity == 'z displacement':
        title = 'Z Displacement'
        sub_plot_titles = ('Interpolated Z Displacement', 'Original Z Displacement')
        color_quantity = grid_FEM[:,2]
        global_min = min(grid_FEM[:,2].min(),original_FEM[:,2].min())
        global_max = max(grid_FEM[:,2].max(),original_FEM[:,2].max())
        index = 2

    else:
        raise ValueError("Invalid plot quantity. Choose from: 'stress', 'x displacement', 'y displacement', 'z displacement'")
                             
    grid_points = point_coordinates
    
    # Extract the x,y,z coordinates from the FEM data
    x_values = FEM_data[:,2]
    y_values = FEM_data[:,3]
    z_values = FEM_data[:,4]
    xyz_FEM = np.column_stack((x_values, y_values, z_values)) 
    original_points = xyz_FEM

    # Interpolated FEM data
    tolerance = 1e-6  
    inside_mask = sdf <= tolerance
    grid_FEM_filtered  = grid_FEM[inside_mask]
    grid_points_filtered = grid_points[inside_mask]
    color_quantity_filtered = color_quantity[inside_mask]


    # --- Create subplots ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles= sub_plot_titles,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    # First subplot 
    fig.add_trace(
        go.Scatter3d(
            x=grid_points_filtered[:,0],
            y=grid_points_filtered[:,1],
            z=grid_points_filtered[:,2],
            mode='markers',
            marker=dict(
                size=2,
                color=color_quantity_filtered,  
                colorscale='Viridis',
                cmin=global_min,
                cmax=global_max,
                opacity=1,
                colorbar=dict(
                    title= title,
                    thickness=20,
                    x=1.1, 
                    y=0.5,
                    len=0.7
                )
            )
        ),
        row=1, col=1
    )

    # Second subplot 
    fig.add_trace(
        go.Scatter3d(
            x= original_points[:,0],
            y= original_points[:,1],
            z= original_points[:,2],
            mode='markers',
            marker=dict(
                size=2,
                color= original_FEM[:,index],  
                colorscale='Viridis',
                cmin=global_min,
                cmax=global_max,
                opacity=1,
                showscale=False
            )
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text='FEM Results: Interpolated vs Original',
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate',
            aspectmode='data',
            xaxis=dict(backgroundcolor='white'),
            yaxis=dict(backgroundcolor='white'),
            zaxis=dict(backgroundcolor='white'),
        ),
        scene2=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate',
            aspectmode='data',
            xaxis=dict(backgroundcolor='white'),
            yaxis=dict(backgroundcolor='white'),
            zaxis=dict(backgroundcolor='white'),
        )
    )

    return fig 
# -------------------------------------------------------------------------------------------------------------



# Function 2 --> Plot FEM data -> All points except zero values are plotted 
def plot_fem_data_without_SDF_masking(
    FEM_data,
    point_coordinates,
    grid_FEM,
    plot_quantity,
    load_case
):
    # 1) Select the values based on the load case  
    if load_case == 'horizontal':
        x_disp = FEM_data[:, 10]
        y_disp = FEM_data[:, 11]
        z_disp = FEM_data[:, 12]
        stress = FEM_data[:, 14]
        original_FEM = np.column_stack((x_disp, y_disp, z_disp, stress))
    elif load_case == 'vertical':
        x_disp = FEM_data[:,  5]
        y_disp = FEM_data[:,  6]
        z_disp = FEM_data[:,  7]
        stress = FEM_data[:,  9]
        original_FEM = np.column_stack((x_disp, y_disp, z_disp, stress))
    elif load_case == 'diagonal':
        x_disp = FEM_data[:, 15]
        y_disp = FEM_data[:, 16]
        z_disp = FEM_data[:, 17]
        stress = FEM_data[:, 19]
        original_FEM = np.column_stack((x_disp, y_disp, z_disp, stress))
    elif load_case == 'torsional':
        x_disp = FEM_data[:, 20]
        y_disp = FEM_data[:, 21]
        z_disp = FEM_data[:, 22]
        stress = FEM_data[:, 24]
        original_FEM = np.column_stack((x_disp, y_disp, z_disp, stress))
    else:
        raise ValueError(
            "Invalid load case. "
            "Choose from 'horizontal', 'vertical', 'diagonal', or 'torsional'."
        )

    # 2) format plot based on the selected quantity
    if plot_quantity == 'stress':
        title = 'Von-Mises Stress'
        sub_plot_titles = ('Interpolated Von-Mises Stress', 'Original Von-Mises Stress')
        color_quantity = grid_FEM[:, 3]
        qcol = 3
        global_min = min(grid_FEM[:, 3].min(), original_FEM[:, 3].min())
        global_max = max(grid_FEM[:, 3].max(), original_FEM[:, 3].max())

    elif plot_quantity == 'x displacement':
        title = 'X Displacement'
        sub_plot_titles = ('Interpolated X Displacement', 'Original X Displacement')
        color_quantity = grid_FEM[:, 0]
        qcol = 0
        global_min = min(grid_FEM[:, 0].min(), original_FEM[:, 0].min())
        global_max = max(grid_FEM[:, 0].max(), original_FEM[:, 0].max())

    elif plot_quantity == 'y displacement':
        title = 'Y Displacement'
        sub_plot_titles = ('Interpolated Y Displacement', 'Original Y Displacement')
        color_quantity = grid_FEM[:, 1]
        qcol = 1
        global_min = min(grid_FEM[:, 1].min(), original_FEM[:, 1].min())
        global_max = max(grid_FEM[:, 1].max(), original_FEM[:, 1].max())

    elif plot_quantity == 'z displacement':
        title = 'Z Displacement'
        sub_plot_titles = ('Interpolated Z Displacement', 'Original Z Displacement')
        color_quantity = grid_FEM[:, 2]
        qcol = 2
        global_min = min(grid_FEM[:, 2].min(), original_FEM[:, 2].min())
        global_max = max(grid_FEM[:, 2].max(), original_FEM[:, 2].max())

    else:
        raise ValueError(
            "Invalid plot quantity. "
            "Choose from: 'stress', 'x displacement', 'y displacement', 'z displacement'."
        )

    grid_points = point_coordinates

    # Extract the x,y,z coordinates from the FEM data
    x_values = FEM_data[:, 2]
    y_values = FEM_data[:, 3]
    z_values = FEM_data[:, 4]
    original_points = np.column_stack((x_values, y_values, z_values))


    interp_mask = (color_quantity != 0)
    grid_FEM_filtered       = grid_FEM[interp_mask]
    grid_points_filtered    = grid_points[interp_mask]
    color_quantity_filtered = color_quantity[interp_mask]

    orig_vals = original_FEM[:, qcol]
    orig_mask = (orig_vals != 0)
    original_points_filtered = original_points[orig_mask]
    original_FEM_filtered    = original_FEM[orig_mask]

    # --- Create subplots ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=sub_plot_titles,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    # First subplot 
    fig.add_trace(
        go.Scatter3d(
            x=grid_points_filtered[:, 0],
            y=grid_points_filtered[:, 1],
            z=grid_points_filtered[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color_quantity_filtered,
                colorscale='Viridis',
                cmin=global_min,
                cmax=global_max,
                opacity=1,
                colorbar=dict(
                    title=title,
                    thickness=20,
                    x=1.1,
                    y=0.5,
                    len=0.7
                )
            )
        ),
        row=1, col=1
    )

    # Second subplot 
    fig.add_trace(
        go.Scatter3d(
            x=original_points_filtered[:, 0],
            y=original_points_filtered[:, 1],
            z=original_points_filtered[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=original_FEM_filtered[:, qcol],
                colorscale='Viridis',
                cmin=global_min,
                cmax=global_max,
                opacity=1,
                showscale=False
            )
        ),
        row=1, col=2
    )

    # --- Update layout for 3D scenes ---
    fig.update_layout(
        title_text='FEM Results: Interpolated vs Original',
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate',
            aspectmode='data',
            xaxis=dict(backgroundcolor='white'),
            yaxis=dict(backgroundcolor='white'),
            zaxis=dict(backgroundcolor='white'),
        ),
        scene2=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate',
            aspectmode='data',
            xaxis=dict(backgroundcolor='white'),
            yaxis=dict(backgroundcolor='white'),
            zaxis=dict(backgroundcolor='white'),
        )
    )

    return fig