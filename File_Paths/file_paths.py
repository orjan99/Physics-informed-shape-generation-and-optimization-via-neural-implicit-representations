import os 

# Root directory of your repo (assumes this file is in the root or near it)
#REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) 
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))     

# Utility to make sub-paths
def repo_path(*path_parts):
    return os.path.join(REPO_ROOT, *path_parts)

# Training log directories
threeD_PINN_Training_Log_Dir = repo_path("Training_Log", "PINN_training", "3D_PINN")
twoD_PINN_Training_Log_Dir   = repo_path("Training_Log", "PINN_training", "2D_PINN")
threeD_GINN_Training_Log_Dir = repo_path("Training_Log", "GINN_training", "3D_GINN")
twoD_GINN_Training_Log_Dir   = repo_path("Training_Log", "GINN_training", "2D_GINN")
Bridge_Training_Log_Dir      = repo_path("Training_Log", "PINN_training", "2D_PINN", "Bridge") 
Cantilever_beam_Training_Log_Dir = repo_path("Training_Log", "PINN_training", "2D_PINN", "Cantilever_Beam") 

# Trained models
threeD_PINN_trained_model_dir = repo_path("Trained_Models", "PINN_Models", "3D_PINN")
twoD_PINN_trained_model_dir   = repo_path("Trained_Models", "PINN_Models", "2D_PINN")
threeD_GINN_trained_model_dir = repo_path("Trained_Models", "GINN_Models", "3D_GINN")
twoD_GINN_trained_model_dir   = repo_path("Trained_Models", "GINN_Models", "2D_GINN")
Bridge_trained_model_dir      = repo_path("Trained_Models", "PINN_Models", "2D_PINN", "Bridge")
Cantilever_beam_trained_model_dir = repo_path("Trained_Models", "PINN_Models", "2D_PINN", "Cantilever_Beam") 

# SimJEB data paths
mesh_path         = repo_path("SimJEB_Data", "SimJEB_surfmesh_(obj)")
FEM_path          = repo_path("SimJEB_Data", "SimJEB _FEM_Results")
point_cloud_path  = repo_path("SimJEB_Data", "Point_Cloud_Dataset")
volume_mesh_path  = repo_path("SimJEB_Data", "SimJEB_volmesh_(vtk)")
interfaces_path   = repo_path("SimJEB_Data", "Interfaces") 
step_file_path    = repo_path("SimJEB_Data", "SimJEB Step_Files")
meta_data_path    = repo_path("SimJEB_Data", "Meta data")
data_path = repo_path("SimJEB_Data")


# Final Trained Models
BRIDGE_PINN_Path = repo_path("Trained_Models_final", "PINN","2D_PINN") 
BRIDGE_GINN_path = repo_path("Trained_Models_final", "GINN","2D_GINN") 

