import numpy as np
import plotly.graph_objects as go # type: ignore

# Function to plot grid points and mesh 
def plot_grid_points(mesh, grid_points):


    verts = mesh.vertices
    faces = mesh.faces
    xg, yg, zg = grid_points[:,0], grid_points[:,1], grid_points[:,2]

    # 1) mesh 
    mesh_trace = go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color='red',
        opacity=0.9,
        flatshading=True
    )

    # 2) grid points
    scatter_trace = go.Scatter3d(
        x=xg, y=yg, z=zg,
        mode='markers',
        marker=dict(
            size=2,
            color=zg,           
            colorscale='Viridis',
            opacity=0.7
        )
    )

    fig = go.Figure(data=[mesh_trace, scatter_trace])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'    
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Mesh + Grid‐Point Cloud"
    )
    return fig