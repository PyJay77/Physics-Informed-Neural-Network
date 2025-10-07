import torch
from model_unsteady_optimized import NS
from model_optimized import PINN
from points_generator_optimized import generator

class Loss():
    def __init__(self,
                x_domain:torch.Tensor,
                y_domain:torch.Tensor,
                t_domain:torch.Tensor,
                circle_coordinates:torch.Tensor,
                properties:torch.Tensor,
                U_old:torch.Tensor,
                time_parameters:torch.Tensor,
                n_points:torch.Tensor,
                number_intervals:torch.Tensor, 
                other_hyperparameters:torch.Tensor,
                device= None):

        '''
        This class implements the loss function for the unsteady Navier-Stokes equations.
        x_domain: tensor of shape (2) representing the x domain [x_min, x_max]
        y_domain: tensor of shape (2) representing the y domain [y_min, y_max]
        t_domain: tensor of shape (2) representing the t domain [t_min, t_max]
        circle_cordinates: tensor of shape (3) representing the circle coordinates [x_center, y_center, radius]
        properties: tensor of shape (3) representing the properties [rho, nu, U_max]
        U_old: tensor of shape (1) representing the old maximum velocity
        time_parameters: tensor of shape (2) representing the time parameters  [dt,tau] - NOTE: dt and tau are not dimensionless
        n_points: tensor of shape (2) representing the number of points [n_interior, n_boundary]
        number_intervals: tensor of shape (1) representing the number of time intervals
        other_hyperparameters: tensor of shape (2) representing the properties [epsilon, lambda_frequency,alpha]
        device: device to run the model on (cpu or cuda)
        '''

        self.Lx = (x_domain[1] - x_domain[0]).to(device)
        self.Ly = (y_domain[1] - y_domain[0]).to(device)
        self.T = (t_domain[1] - t_domain[0]).to(device)
        self.xc = circle_coordinates[0].to(device)
        self.yc = circle_coordinates[1].to(device)
        self.r = circle_coordinates[2].to(device)
        self.D = (2 * self.r).to(device)
        self.rho = properties[0].to(device)
        self.nu = properties[1].to(device)
        self.U_max = properties[2].to(device)
        self.Re = (self.U_max * self.D / self.nu).to(device)
        self.U_max_old = U_old.to(device)
        self.dt = (time_parameters[0] * self.U_max / self.D).to(device)
        self.tau = (time_parameters[1] * self.U_max / self.D).to(device)
        self.epsilon = other_hyperparameters[0].to(device)
        self.updating_lambda_frequency = other_hyperparameters[1].to(device)
        self.alpha = other_hyperparameters[2].to(device)
        self.n_interior = n_points[0].to(device)
        self.n_boundary = n_points[1].to(device)
        self.number_intervals = number_intervals.to(device)
        self.eps = torch.tensor(1e-8,device=device)
        self.device = device

    def df(self,function,variable):
        return torch.autograd.grad(function, variable, grad_outputs=torch.ones_like(function), create_graph=True, retain_graph=True)[0]

    def compute_grad_norm(self,pinn:NS,target_loss):
        # Let set to zero the gradients of the parameters
        pinn.zero_grad()
        # Compute the gradients of the loss with respect to the parameters
        target_loss.backward(retain_graph=True)
        # Compute the L2 norm of the gradients
        grads = [param.grad.flatten() for param in pinn.parameters() if param.grad is not None]
        grad_tensor = torch.cat(grads,dim=0)
        grad_norm = torch.norm(grad_tensor, p=2)
        
        return grad_norm.detach()

    def PDE(self,pinn:NS,generator:generator,epoch:int):
        # Generate points
        x_in, y_in, t_in = generator.generate_interior_points()

        x_in = x_in.to(self.device).requires_grad_(True)
        y_in = y_in.to(self.device).requires_grad_(True)
        t_in = t_in.to(self.device).requires_grad_(True)

        Rx_list = []; Ry_list = []; Rc_list = []

        for n in range(self.number_intervals):
            # Predicting u,v,p
            t_n = t_in[:,n:n+1]
            u,v,p = pinn(x_in,y_in,t_n)

            u_t = self.df(u,t_n)
            u_x = self.df(u,x_in)
            u_y = self.df(u,y_in)
            u_xx = self.df(u_x,x_in)
            u_yy = self.df(u_y,y_in)

            v_t = self.df(v,t_n)
            v_x = self.df(v,x_in)
            v_y = self.df(v,y_in)
            v_xx = self.df(v_x,x_in)
            v_yy = self.df(v_y,y_in)

            p_x = self.df(p,x_in)
            p_y = self.df(p,y_in)

            # Residuals of the equations
            Rx = u_t + (u*u_x + v*u_y) + p_x - (u_xx + u_yy).div(self.Re)            # N x 1
            Ry = v_t + (u*v_x + v*v_y) + p_y - (v_xx + v_yy).div(self.Re)            # N x 1
            Rc = u_x + v_y                                                           # N x 1

            Rx_list.append(Rx.pow(2).mean().unsqueeze(0)); Ry_list.append(Ry.pow(2).mean().unsqueeze(0)); Rc_list.append(Rc.pow(2).mean().unsqueeze(0))

        Rx_all = torch.stack(Rx_list,dim=1)                                          # 1 x number_intervals
        Ry_all = torch.stack(Ry_list,dim=1)                                          # 1 x number_intervals
        Rc_all = torch.stack(Rc_list,dim=1)                                          # 1 x number_intervals


        Loss_residual = Rx_all + Ry_all + Rc_all                                                # 1 x number_intervals

        # Computing casual weights
        with torch.no_grad():
            cumulative_loss = torch.cumsum(Loss_residual.detach(),dim=1)                        # 1 x number_intervals
            cumulative_loss = torch.cat([torch.tensor([0],device=self.device),cumulative_loss[0,:-1]])
            W = (torch.exp(-self.epsilon * cumulative_loss)).view(1,-1)
        
        if epoch% self.updating_lambda_frequency == 0:
            print(f'Casuality weights: {[f"{weight.item():.3f}" for weight in W.squeeze()]}')

        loss_Rx = Rx_all.mul(W).mean()
        loss_Ry = Ry_all.mul(W).mean()
        loss_Rc = Rc_all.mul(W).mean()

        return loss_Rx, loss_Ry, loss_Rc

    
    def initial_loss(self,pinn:NS, pinn_initial:PINN,generator:generator):
        # Generate points
        x_i,y_i,t_i =generator.generate_initial_points()
        x_i = x_i.to(self.device)
        y_i = y_i.to(self.device)
        t_i = t_i.to(self.device)

        u,v,p = pinn(x_i, y_i, t_i)
        u_initial, v_initial, p_initial = pinn_initial(x_i, y_i)

        # Fixing scale differences
        u_initial = u_initial.mul(self.U_max_old/self.U_max)
        v_initial = v_initial.mul(self.U_max_old/self.U_max)
        p_initial = p_initial.mul(self.U_max_old**2/self.U_max**2)
        # Computing losses
        loss_u0 = (u - u_initial).pow(2).mean()
        loss_v0 = (v - v_initial).pow(2).mean()
        loss_p0 = (p - p_initial).pow(2).mean()
   
        return loss_u0, loss_v0, loss_p0

    def boundary_loss(self,pinn:NS,generator:generator):
        # Generate points
        down, up, left, right = generator.generate_boundary_points()
        # Down
        down = down.to(self.device) 
        xx_d = down[:,0:1]; yy_d = down[:,1:2]; tt_boundary = down[:,2:3]
        # Up
        up = up.to(self.device)
        xx_u = up[:,0:1]; yy_u = up[:,1:2];              
        # Left  
        left = left.to(self.device)
        xx_l = left[:,0:1]; yy_l = left[:,1:2]
        # Right
        right = right.to(self.device) 
        xx_r = right[:,0:1]; yy_r = right[:,1:2]
        xx_r = xx_r.requires_grad_(True)
        yy_r = yy_r.requires_grad_(True)
        # Circle
        xx_c, yy_c, _ = generator.generate_circle_points() 
        xx_c = xx_c.to(self.device); yy_c = yy_c.to(self.device)

        # Computing the values
        u_d, v_d, p_d = pinn(xx_d,yy_d,tt_boundary) 
        u_u, v_u, p_u = pinn(xx_u,yy_u,tt_boundary)
        u_l, v_l, p_l = pinn(xx_l,yy_l,tt_boundary)
        u_r, v_r, p_r = pinn(xx_r,yy_r,tt_boundary)
        u_c, v_c, p_c = pinn(xx_c,yy_c,tt_boundary)

        # Inlet ----> parabolic velocity profile
        u1 = 4 * self.D / self.Ly * (yy_l - self.D/self.Ly * yy_l**2)
        u0 = 4 * self.D / self.Ly * (self.U_max_old/self.U_max) * (yy_l - self.D/self.Ly * yy_l**2)
        u_in = u1 + (u0 - u1).mul(torch.exp(- tt_boundary / self.tau))
        loss_l = (u_l - u_in).pow(2).mean() + v_l.pow(2).mean()

        # Outlet ----> do-nothing condition
        u_rx = self.df(u_r,xx_r)
        v_rx = self.df(v_r,xx_r)
        u_ry = self.df(u_r,yy_r)
        loss_r = (2/self.Re * u_rx - p_r).pow(2).mean() + (u_ry + v_rx).pow(2).mean()

        # No-slip conditions
        loss_c = (u_c.pow(2) + v_c.pow(2)).mean()
        loss_d = (u_d.pow(2) + v_d.pow(2)).mean()
        loss_u = (u_u.pow(2) + v_u.pow(2)).mean()

        return loss_c, loss_d, loss_u, loss_l, loss_r

    def __call__(self,pinn:NS,pinn_initial:PINN,weights:torch.Tensor,epoch,generator:generator):

        w = weights.to(self.device)

        loss_Rx, loss_Ry, loss_R_continuity = self.PDE(pinn,generator,epoch)
        loss_u0, loss_v0, loss_p0 = self.initial_loss(pinn,pinn_initial,generator)
        loss_c, loss_d, loss_u, loss_l, loss_r = self.boundary_loss(pinn,generator)

        total_loss = (w[0]*loss_Rx + w[1]*loss_Ry + w[2]*loss_R_continuity +
                    w[3]*loss_u0 + w[4]*loss_v0 + w[5]*loss_p0 +
                    w[6]*loss_c  + w[7]*loss_d  + w[8]*loss_u + w[9]*loss_l + w[10]*loss_r)

        loss_PDE     = loss_Rx + loss_Ry + loss_R_continuity
        loss_initial = loss_u0 + loss_v0 + loss_p0
        loss_BC      = loss_c + loss_d + loss_u + loss_l + loss_r

        if epoch % self.updating_lambda_frequency == 0 and epoch != 0:

            norm_Rx  = self.compute_grad_norm(pinn, loss_Rx)
            norm_Ry  = self.compute_grad_norm(pinn, loss_Ry)
            norm_Rc  = self.compute_grad_norm(pinn, loss_R_continuity)
            norm_u0  = self.compute_grad_norm(pinn, loss_u0)
            norm_v0  = self.compute_grad_norm(pinn, loss_v0)
            norm_p0  = self.compute_grad_norm(pinn, loss_p0)
            norm_cg  = self.compute_grad_norm(pinn, loss_c)
            norm_dg  = self.compute_grad_norm(pinn, loss_d)
            norm_ug  = self.compute_grad_norm(pinn, loss_u)
            norm_lg  = self.compute_grad_norm(pinn, loss_l)
            norm_rg  = self.compute_grad_norm(pinn, loss_r)

            # Corretto calcolo di S
            S = (norm_Rx + norm_Ry + norm_Rc + norm_u0 + norm_v0 + norm_p0 +
                norm_cg + norm_dg + norm_ug + norm_lg + norm_rg)

            # λ_hat (numeri freddi, perché S e norm_* sono detach)
            lambda_Rx_hat = S / (norm_Rx + self.eps)
            lambda_Ry_hat = S / (norm_Ry + self.eps)
            lambda_Rc_hat = S / (norm_Rc + self.eps)
            lambda_u0_hat = S / (norm_u0 + self.eps)
            lambda_v0_hat = S / (norm_v0 + self.eps)
            lambda_p0_hat = S / (norm_p0 + self.eps)
            lambda_c_hat  = S / (norm_cg + self.eps)
            lambda_d_hat  = S / (norm_dg + self.eps)
            lambda_u_hat  = S / (norm_ug + self.eps)
            lambda_l_hat  = S / (norm_lg + self.eps)
            lambda_r_hat  = S / (norm_rg + self.eps)

            # EMA con detach a valle (comunque siamo in no_grad)
            lambda_Rx = self.alpha * w[0] + (1 - self.alpha) * lambda_Rx_hat
            lambda_Ry = self.alpha * w[1] + (1 - self.alpha) * lambda_Ry_hat
            lambda_Rc = self.alpha * w[2] + (1 - self.alpha) * lambda_Rc_hat
            lambda_u0 = self.alpha * w[3] + (1 - self.alpha) * lambda_u0_hat
            lambda_v0 = self.alpha * w[4] + (1 - self.alpha) * lambda_v0_hat
            lambda_p0 = self.alpha * w[5] + (1 - self.alpha) * lambda_p0_hat
            lambda_c  = self.alpha * w[6] + (1 - self.alpha) * lambda_c_hat
            lambda_d  = self.alpha * w[7] + (1 - self.alpha) * lambda_d_hat 
            lambda_u  = self.alpha * w[8] + (1 - self.alpha) * lambda_u_hat 
            lambda_l  = self.alpha * w[9] + (1 - self.alpha) * lambda_l_hat
            lambda_r  = self.alpha * w[10]+ (1 - self.alpha) * lambda_r_hat 

            weights = torch.tensor([lambda_Rx, lambda_Ry, lambda_Rc,
                                lambda_u0, lambda_v0, lambda_p0,
                                lambda_c, lambda_d, lambda_u, lambda_l, lambda_r], device=self.device)

            print(f'Lambda weights: {[f"{weight:.3f}" for weight in weights.tolist()]}')

        return  total_loss, loss_PDE, loss_BC,loss_initial, weights      
    






