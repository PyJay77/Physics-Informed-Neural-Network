import torch
import math

class generator():
    def __init__(self,
                x_domain:torch.Tensor,
                y_domain:torch.Tensor,
                t_domain:torch.Tensor,
                circle_coordinate:torch.Tensor,
                distance: torch.Tensor,
                properties:torch.Tensor,
                number_intervals:torch.Tensor,
                n_points:torch.Tensor
                ):

        '''
        x_domain: tensor of shape (2) representing the x domain [x_min, x_max]
        y_domain: tensor of shape (2) representing the y domain [y_min, y_max]
        t_domain: tensor of shape (2) representing the t domain [t_min, t_max]
        circle_coordinate: tensor of shape (3) representing the circle coordinates [x_center, y_center, radius]
        properties: tensor of shape (4) representing the properties [rho, nu, U_max]
        number_intervals: tensor of shape (1) representing the number of time intervals we want to use to discretize the time domain and sample in each interval
        n_points: tensor of shape (2) representing the number of interior and boundary points to be sampled, defined as [n_interior, n_boundary]
        '''
        self.Lx = x_domain[1] - x_domain[0]
        self.Ly = y_domain[1] - y_domain[0]
        self.T = t_domain[1] - t_domain[0]
        self.xc, self.yc, self.r = circle_coordinate[0], circle_coordinate[1], circle_coordinate[2]
        self.distance = distance
        self.D = 2*self.r
        self.rho, self.nu, self.U_max = properties[0], properties[1], properties[2]
        self.n_interior, self.n_boundary = n_points[0], n_points[1]
        self.number_intervals = number_intervals[0]
        self.dt = self.T / self.number_intervals
        self.pi = torch.tensor(math.pi)

    def generate_interior_points(self):

        '''
        distance: tensor of shape (1) representing the distance from the center of the circle to the center of the wake region where we want to sample more points
        '''

        x_int = torch.rand(size=(self.n_interior,1)) * self.Lx                             # n_interior x 1
        y_int = torch.rand(size=(self.n_interior,1)) * self.Ly                             # n_interior x 1

        # Mask to exclude points inside the circle
        mask_int = (x_int - self.xc)**2 + (y_int - self.yc)**2 > self.r**2                 # n_interior x 1

        x_int = x_int[mask_int].unsqueeze(1)                                               # n_filtered x 1
        y_int = y_int[mask_int].unsqueeze(1)                                               # n_filtered x 1

        x_wake = torch.rand(size=(self.n_interior,1)) * self.Lx                            # n_interior x 1
        y_wake = (self.yc - self.distance) + 2*self.distance * torch.rand(size=(self.n_interior,1))  # n_interior x 1

        # Mask to exclude points inside the circle
        mask_wake = (x_wake - self.xc)**2 + (y_wake - self.yc)**2 > self.r**2              # n_interior x 1

        x_wake = x_wake[mask_wake].unsqueeze(1)                                            # n_filtered x 1
        y_wake = y_wake[mask_wake].unsqueeze(1)                                            # n_filtered x 1

        xx = (torch.cat([x_int,x_wake],dim=0)).div(self.D)                                 # n_total x 1
        yy = (torch.cat([y_int,y_wake],dim=0)).div(self.D)                                 # n_total x 1

        dt_tensor = torch.rand(size=(xx.shape[0],self.number_intervals)) * self.dt         # n_total x number_intervals
        offsets = torch.arange(self.number_intervals).reshape(1, -1) * self.dt             # 1 x number_intervals
        tt = (dt_tensor + offsets).mul(self.U_max/self.D)                                  # n_total x number_intervals

        return xx,yy,tt                                                                    # ADIMENSIONALIZED COORDINATES

    def generate_initial_points(self):

        x_int = torch.rand(size=(self.n_interior,1)) * self.Lx                             # n_interior x 1
        y_int = torch.rand(size=(self.n_interior,1)) * self.Ly                             # n_interior x 1

        # Mask to exclude points inside the circle
        mask_int = (x_int - self.xc)**2 + (y_int - self.yc)**2 > self.r**2                 # n_interior x 1

        xx = (x_int[mask_int].unsqueeze(1)).div(self.D)                                    # n_filtered x 1
        yy = (y_int[mask_int].unsqueeze(1)).div(self.D)                                    # n_filtered x 1
        tt = torch.zeros(size=(xx.shape[0],1))                                             # n_filtered x 1

        return xx,yy,tt                                                                    # ADIMENSIONALIZED COORDINATES

    def generate_circle_points(self):

        theta = 2*self.pi*torch.rand(size=(self.n_boundary,1))                             # n_boundary x 1
        xx = (self.xc + self.r*torch.cos(theta)).div(self.D)                               # n_boundary x 1
        yy = (self.yc + self.r*torch.sin(theta)).div(self.D)                               # n_boundary x 1                                                    
        tt = (torch.rand(size=(self.n_boundary,1))).mul(self.T * self.U_max / self.D)      # n_boundary x 1                                                  

        return xx,yy,tt                                                                    # ADIMENSIONALIZED COORDINATES

    def generate_boundary_points(self):
                
        x_bnd = (torch.rand(size=(self.n_boundary,1))).mul(self.Lx / self.D)               # n_boundary x 1
        y_bnd = (torch.rand(size=(self.n_boundary,1))).mul(self.Ly / self.D)               # n_boundary x 1
        tt = (torch.rand(size=(self.n_boundary,1))).mul(self.T * self.U_max / self.D)      # n_boundary x 1

        x_left = torch.zeros(size=(self.n_boundary,1))                                     # n_boundary x 1
        x_right = torch.full_like(x_bnd,self.Lx/self.D)                                    # n_boundary x 1
        y_bottom = torch.zeros(size=(self.n_boundary,1))                                   # n_boundary x 1
        y_top = torch.full_like(y_bnd,self.Ly/self.D)                                      # n_boundary x 1

        down = torch.cat([x_bnd,y_bottom,tt],dim=1)                                        # n_boundary x 3
        up = torch.cat([x_bnd,y_top,tt],dim=1)                                             # n_boundary x 3
        left = torch.cat([x_left,y_bnd,tt],dim=1)                                          # n_boundary x 3
        right = torch.cat([x_right,y_bnd,tt],dim=1)                                        # n_boundary x 3

        return down,up,left,right                                                          # ADIMENSIONALIZED COORDINATES


