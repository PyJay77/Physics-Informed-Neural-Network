import torch
from torch import nn
from typing import Tuple
import math

class PINN(nn.Module):
    def __init__(self,
                 network_parameters:torch.Tensor,
                 fourier_features_parameters:torch.Tensor,
                 x_domain:torch.Tensor,
                 y_domain:torch.Tensor,
                 circle_coordinates:torch.Tensor,
                 properties:torch.Tensor,
                 random_weight_initialization_parameters:torch.Tensor,
                 device = None):
        super().__init__()

        '''
        network_parameters: tensor of shape (4) representing the network parameters [num_input, num_output, num_layers, num_neurons]
        fourier_features_parameters: tensor of shape (3) representing the Fourier Features parameters [sigma_x, sigma_y, sigma_t]
        x_domain: tensor of shape (2) representing the x domain [x_min, x_max]
        y_domain: tensor of shape (2) representing the y domain [y_min, y_max]
        circle_cordinates: tensor of shape (3) representing the circle coordinates [x_center, y_center, radius]
        properties: tensor of shape (3) representing the properties [rho, nu, U_max]
        random_weight_initialization_parameters: tensor of shape (2) representing the random weight initialization parameters [mu, sigma]
        device: device to run the model on (cpu or cuda)
        N = number of input points
        '''

        # Initializing parameters
        self.num_input, self.num_output, self.num_layers, self.num_neurons = network_parameters[0], network_parameters[1], network_parameters[2], network_parameters[3]
        self.M = self.num_neurons // (self.num_input * 2)
        self.sigma_x, self.sigma_y = fourier_features_parameters[0], fourier_features_parameters[1]
        self.Lx = x_domain[1] - x_domain[0]
        self.Ly = y_domain[1] - y_domain[0]
        self.xc, self.yc, self.r = circle_coordinates[0], circle_coordinates[1], circle_coordinates[2]
        self.D = 2*self.r
        self.rho, self.nu, self.U_max = properties[0], properties[1], properties[2]
        self.x_max = self.Lx / self.D
        self.y_max = self.Ly / self.D
        self.mu, self.sigma = random_weight_initialization_parameters[0], random_weight_initialization_parameters[1]
        self.device = device

        # Fourier Features Matrix Initialization
        Wx = (torch.randn(size=(1,self.M))).mul(self.sigma_x)
        Wy = (torch.randn(size=(1,self.M))).mul(self.sigma_y)
        self.register_buffer("Wx",Wx)
        self.register_buffer("Wy",Wy)

        # Activation function
        self.act = nn.Tanh()

        # Initial Matrix Initialization
        self.U = nn.Parameter(torch.empty(size=(self.num_neurons,self.num_neurons)))                                  # num_neurons x num_neurons
        self.V = nn.Parameter(torch.empty(size=(self.num_neurons,self.num_neurons)))                                  # num_neurons x num_neurons

        self.sU = nn.Parameter(torch.empty(size=(self.num_neurons,1)))                                                # num_neurons x 1
        self.sV = nn.Parameter(torch.empty(size=(self.num_neurons,1)))                                                # num_neurons x 1

        self.bU = nn.Parameter(torch.zeros(size=(self.num_neurons,1)))                                                # num_neurons x 1
        self.bV = nn.Parameter(torch.zeros(size=(self.num_neurons,1)))                                                # num_neurons x 1

        # Input Layer Matrix Initialization
        self.W_in = nn.Parameter(torch.empty(size=(self.num_input * 2 * self.M,self.num_neurons)))                    # (num_input * 2 * M) x num_neurons
        self.s_in = nn.Parameter(torch.empty(size=(self.num_neurons,1)))                                              # num_neurons x 1
        self.b_in = nn.Parameter(torch.zeros(size=(self.num_neurons,1)))                                              # num_neurons x 1

        # Hidden Layers Matrix Initialization
        self.W_h = nn.ParameterList(
            [nn.Parameter(torch.empty(size=(self.num_neurons,self.num_neurons))) for _ in range(self.num_layers - 1)] # num_neurons x num_neurons
        )

        self.s_h = nn.ParameterList(
            [nn.Parameter(torch.empty(size=(self.num_neurons,1))) for _ in range(self.num_layers - 1)]                # num_neurons x 1
        )
        
        self.b_h = nn.ParameterList(
            [nn.Parameter(torch.zeros(size=(self.num_neurons,1))) for _ in range(self.num_layers - 1)]                # num_neurons x 1
        )

        # Output Layer Matrix Initialization
        self.W_out = nn.Parameter(torch.empty(size=(self.num_neurons,self.num_output)))                               # num_neurons x num_output
        self.s_out = nn.Parameter(torch.empty(size=(self.num_neurons,1)))                                             # num_neurons x 1
        self.b_out = nn.Parameter(torch.zeros(size=(self.num_output,1)))                                              # num_output x 1

        # Weights Initialization
        self._initialize_weights()
        self.to(self.device)

    
    def _initialize_weights(self):

        gain_tanh = nn.init.calculate_gain('tanh')

        # Helper function for Xavier/Glorot initialization
        def _init_weight(param, gain = None):
            if gain:
                nn.init.xavier_uniform_(param, gain=gain)
            else:
                nn.init.xavier_uniform_(param)

        def _init_s(param):
            param.normal_(mean=self.mu, std=self.sigma)
            
        def _apply_scaling(param, s):
            param.data.mul_(torch.exp(-s))

        # Initializing all weights
        with torch.no_grad():

            # Initializing U and V
            _init_weight(self.U, gain=gain_tanh)
            _init_weight(self.V, gain=gain_tanh)
            _init_s(self.sU)
            _init_s(self.sV)

            # Initializing Input Layer
            _init_weight(self.W_in, gain=gain_tanh)
            _init_s(self.s_in)

            # Initializing Hidden Layers
            for W, s in zip(self.W_h, self.s_h):
                _init_weight(W, gain=gain_tanh)
                _init_s(s)

            # Initializing Output Layer
            _init_weight(self.W_out)
            _init_s(self.s_out)

            # Applying exponential scaling
            _apply_scaling(self.U, self.sU)
            _apply_scaling(self.V, self.sV)
            _apply_scaling(self.W_in, self.s_in)
            for W, s in zip(self.W_h, self.s_h):
                _apply_scaling(W, s)
            _apply_scaling(self.W_out, self.s_out)

    
    def forward(self, x, y):

        xx = 2*x.div(self.x_max) - 1                                           # [-1,1]
        yy = 2*y.div(self.y_max) - 1                                           # [-1,1]

        # Fourier Features Mapping
        Bx = xx @ self.Wx                                                      # N x M
        By = yy @ self.Wy                                                      # N x M                                                   # N x M

        xyt_mapped = torch.cat([torch.cos(Bx), torch.sin(Bx),
                                torch.cos(By), torch.sin(By)], dim=1)          # N x (2 * number_input * M) 

        # Passing through the network
        W1 = torch.exp(self.sU).mul(self.U)                                   # num_neurons x num_neurons
        W2 = torch.exp(self.sV).mul(self.V)                                   # num_neurons x num_neurons
        W_in = torch.exp(self.s_in).mul(self.W_in)                            # (num_input * 2 * M) x num_neurons
        W_h = [torch.exp(s).mul(W) for s, W in zip(self.s_h, self.W_h)]       # num_neurons x num_neurons
        W_out = torch.exp(self.s_out).mul(self.W_out)                         # num_neurons x num_output

        U = self.act(xyt_mapped @ W1 + self.bU.T)                             # N x num_neurons
        V = self.act(xyt_mapped @ W2 + self.bV.T)                             # N x num_neurons

        f_l = xyt_mapped @ W_in + self.b_in.T                                 # N x num_neurons
        g_l = self.act(f_l).mul(U) + (1 - self.act(f_l)).mul(V)               # N x num_neurons

        for W, b in zip(W_h, self.b_h):
            f_l = g_l @ W + b.T                                               # N x num_neurons
            g_l = self.act(f_l).mul(U) + (1 - self.act(f_l)).mul(V)           # N x num_neurons

        output = g_l @ W_out + self.b_out.T                                   # N x num_output

        u = output[:,0:1]                                                     # N x 1
        v = output[:,1:2]                                                     # N x 1 
        p = output[:,2:3]                                                     # N x 1

        return u,v,p