
import numpy as np
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
import trimesh # type: ignore

# Function to plot the points on the surfae or inside the geometry (SDF < 0)
def SDF_geometry_plot(points,sdf,mesh):
    """
    Visualize the mesh and the points inside the mesh (SDF ≤ 0).
    """

    tolerance = 1e-6
    mask = sdf <= tolerance
    pts_in   = points[mask]
    sdf_in   = sdf[mask]

    print("Number of points inside the mesh: ", len(pts_in))
    print("Number of points outside the mesh: ", len(points) - len(pts_in))
 
    verts = mesh.vertices
    faces = mesh.faces
    mesh_trace = go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color='lightgray',
        opacity=0.2,
        flatshading=True
    )

    scatter_in = go.Scatter3d(
        x=pts_in[:,0],
        y=pts_in[:,1],
        z=pts_in[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color=sdf_in,          
            colorscale='Blues',    
            cmin= sdf.min(),
            cmax=0,
            opacity=0.8
        ),
        name='SDF ≤ 0'
    )

    fig = go.Figure(data=[mesh_trace, scatter_in])
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Mesh with Inside‐Surface Grid Points (SDF ≤ 0)"
    )
    return fig
