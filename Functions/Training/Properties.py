import torch
 
class Properties(object):
    '''
    This loss class is defin
    '''

    def __init__(self,test_case):
        self.interfaces = test_case.interfaces
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.getLoss = torch.nn.MSELoss()
        self.test_case = test_case

        self.elastic_modulus  = test_case.elastic_modulus     # Young's modulus
        self.poisson_ratio    = test_case.poisson_ratio       # Poisson's ratio
        self.material_density = test_case.material_density    # Material density
        self.domain_volume    = test_case.domain_volume      # Domain volume
        self.force_vector = test_case.force_vector_vertical  # Applied load vector

        # Assign the material properties to the device and convert to pytorch tensor
        self.elastic_modulus  = torch.tensor(self.elastic_modulus, device = self.device, dtype = torch.float32)
        self.poisson_ratio    = torch.tensor(self.poisson_ratio, device = self.device, dtype = torch.float32)
        self.material_density = torch.tensor(self.material_density, device = self.device, dtype = torch.float32)
        self.domain_volume    = torch.tensor(self.domain_volume, device = self.device, dtype = torch.float32)
        self.gravitational_acceleration = torch.tensor(test_case.gravitational_acceleration, device = self.device, dtype = torch.float32)
        self.force_vector = torch.tensor(self.force_vector, device = self.device, dtype = torch.float32)
        self.lame_lambda = (self.elastic_modulus * self.poisson_ratio) / (1 - self.poisson_ratio**2).to(self.device)
        self.lame_mu = self.elastic_modulus / (2 * (1 + self.poisson_ratio)).to(self.device)


        # Non Dimensionalize Physics Quantities
        scale =   test_case.domain_scaling_factor
        #self.L0 = 1/scale
        #self.F0 = self.force_vector[1]
        #self.Sigma0 = self.F0/self.L0**2
        #self.Sigma0 = 1
        #self.L0 = 1/scale
        #self.E0 = self.elastic_modulus
        self.P0 = self.material_density
        self.Sigma0 = 1
        self.F0 = 1
        self.L0 = 1
        self.E0 = 1
        self.P0 = 1
        #self.elastic_modulus = self.elastic_modulus / self.E0
        #self.material_density = self.material_density / self.P0
        #self.lame_lambda = self.lame_lambda / self.E0
        #self.lame_mu = self.lame_mu / self.E0
        #self.gravitational_acceleration = self.gravitational_acceleration * (self.L0*self.P0/(self.E0))
        #self.force_vector = self.force_vector / (self.E0*(self.L0**2))

        self.elastic_modulus = self.elastic_modulus / self.Sigma0
        self.material_density = self.material_density
        self.lame_lambda = self.lame_lambda / self.Sigma0
        self.lame_mu = self.lame_mu / self.Sigma0
        self.force_vector = self.force_vector / self.F0

