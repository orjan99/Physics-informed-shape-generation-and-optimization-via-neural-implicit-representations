
import numpy as np
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
import trimesh # type: ignore

# Function to plot the points on the surfae or inside the geometry (density < 0)
def density_geometry_plot(points,density,mesh,isoline=0.5):
    """
    Visualize the mesh and the points inside the mesh (density ≤ 0) 
    """

    tolerance = isoline 
    mask = density >= tolerance
    pts_in   = points[mask]
    density_in   = density[mask]

    print("Number of points inside the mesh: ", len(pts_in))
    print("Number of points outside the mesh: ", len(points) - len(pts_in))
    
    # Prepare the mesh trace
    verts = mesh.vertices
    faces = mesh.faces
    mesh_trace = go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color='red',
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
            color=density_in,         
            colorscale='thermal',    
            cmin=density.min(),
            cmax=0,
            opacity=0.8,
            colorbar=dict(
                title="Density",
                titleside="right",
                tickmode="auto"
            )
        ),
        name='density ≤ 0'
    )

    fig = go.Figure(data=[mesh_trace, scatter_in])
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Mesh with Inside‐Surface Grid Points (density ≤ 0)"
    )
    return fig

