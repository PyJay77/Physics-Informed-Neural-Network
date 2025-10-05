import torch
from model_optimized import PINN
from points_generator_optimized import generator

class Loss():
    def __init__(self,
                x_domain:torch.Tensor,
                y_domain:torch.Tensor,
                circle_coordinates:torch.Tensor,
                properties:torch.Tensor,
                n_points:torch.Tensor,
                other_hyperparameters:torch.Tensor,
                device= None):

        '''
        This class implements the loss function for the unsteady Navier-Stokes equations.
        x_domain: tensor of shape (2) representing the x domain [x_min, x_max]
        y_domain: tensor of shape (2) representing the y domain [y_min, y_max]
        circle_cordinates: tensor of shape (3) representing the circle coordinates [x_center, y_center, radius]
        properties: tensor of shape (3) representing the properties [rho, nu, U_max]
        n_points: tensor of shape (2) representing the number of points [n_interior, n_boundary]
        other_hyperparameters: tensor of shape (2) representing the properties [epsilon, lambda_frequency,alpha]
        device: device to run the model on (cpu or cuda)
        '''

        self.Lx = (x_domain[1] - x_domain[0]).to(device)
        self.Ly = (y_domain[1] - y_domain[0]).to(device)
        self.xc = circle_coordinates[0].to(device)
        self.yc = circle_coordinates[1].to(device)
        self.r = circle_coordinates[2].to(device)
        self.D = (2 * self.r).to(device)
        self.rho = properties[0].to(device)
        self.nu = properties[1].to(device)
        self.U_max = properties[2].to(device)
        self.Re = (self.U_max * self.D / self.nu).to(device)
        self.epsilon = other_hyperparameters[0].to(device)
        self.updating_lambda_frequency = other_hyperparameters[1].to(device)
        self.alpha = other_hyperparameters[2].to(device)
        self.n_interior = n_points[0].to(device)
        self.n_boundary = n_points[1].to(device)
        self.eps = torch.tensor(1e-8,device=device)
        self.device = device

    def df(self,function,variable):
        return torch.autograd.grad(function, variable, grad_outputs=torch.ones_like(function), create_graph=True, retain_graph=True)[0]

    def compute_grad_norm(self,pinn:PINN,target_loss):
        # Let set to zero the gradients of the parameters
        pinn.zero_grad()
        # Compute the gradients of the loss with respect to the parameters
        target_loss.backward(retain_graph=True)
        # Compute the L2 norm of the gradients
        grads = [param.grad.flatten() for param in pinn.parameters() if param.grad is not None]
        grad_tensor = torch.cat(grads,dim=0)
        grad_norm = torch.norm(grad_tensor, p=2)
        
        return grad_norm.detach()

    def PDE(self,pinn:PINN,generator:generator,epoch:int):
        # Generate points
        x_in, y_in, _ = generator.generate_interior_points()

        x_in = x_in.to(self.device).requires_grad_(True)
        y_in = y_in.to(self.device).requires_grad_(True)

        # Predicting u,v,p
        u,v,p = pinn(x_in,y_in)

        u_x = self.df(u,x_in)
        u_y = self.df(u,y_in)
        u_xx = self.df(u_x,x_in)
        u_yy = self.df(u_y,y_in)

        v_x = self.df(v,x_in)
        v_y = self.df(v,y_in)
        v_xx = self.df(v_x,x_in)
        v_yy = self.df(v_y,y_in)

        p_x = self.df(p,x_in)
        p_y = self.df(p,y_in)

        # Residuals of the equations
        Rx = ((u*u_x + v*u_y) + p_x - (u_xx + u_yy).div(self.Re)).pow(2).mean()           
        Ry = ((u*v_x + v*v_y) + p_y - (v_xx + v_yy).div(self.Re)).pow(2).mean()            
        Rc = (u_x + v_y).pow(2).mean()                                                      

        return Rx, Ry, Rc

    def boundary_loss(self,pinn:PINN,generator:generator):
        # Generate points
        down, up, left, right = generator.generate_boundary_points()
        # Down
        down = down.to(self.device) 
        xx_d = down[:,0:1]; yy_d = down[:,1:2]
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
        u_d, v_d, p_d = pinn(xx_d,yy_d) 
        u_u, v_u, p_u = pinn(xx_u,yy_u)
        u_l, v_l, p_l = pinn(xx_l,yy_l)
        u_r, v_r, p_r = pinn(xx_r,yy_r)
        u_c, v_c, p_c = pinn(xx_c,yy_c)

        # Inlet ----> parabolic velocity profile
        u_in = 4 * self.D / self.Ly * (yy_l - self.D/self.Ly * yy_l**2)
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

    def __call__(self,pinn:PINN,weights:torch.Tensor,epoch,generator:generator):

        w = weights.to(self.device)

        loss_Rx, loss_Ry, loss_R_continuity = self.PDE(pinn,generator,epoch)
        loss_c, loss_d, loss_u, loss_l, loss_r = self.boundary_loss(pinn,generator)

        total_loss = (w[0]*loss_Rx + w[1]*loss_Ry + w[2]*loss_R_continuity + w[3]*loss_c  + w[4]*loss_d  + w[5]*loss_u + w[6]*loss_l + w[7]*loss_r)

        loss_PDE     = loss_Rx + loss_Ry + loss_R_continuity
        loss_BC      = loss_c + loss_d + loss_u + loss_l + loss_r

        if epoch % self.updating_lambda_frequency == 0 and epoch != 0:

            norm_Rx  = self.compute_grad_norm(pinn, loss_Rx)
            norm_Ry  = self.compute_grad_norm(pinn, loss_Ry)
            norm_Rc  = self.compute_grad_norm(pinn, loss_R_continuity)
            norm_cg  = self.compute_grad_norm(pinn, loss_c)
            norm_dg  = self.compute_grad_norm(pinn, loss_d)
            norm_ug  = self.compute_grad_norm(pinn, loss_u)
            norm_lg  = self.compute_grad_norm(pinn, loss_l)
            norm_rg  = self.compute_grad_norm(pinn, loss_r)

            # Corretto calcolo di S
            S = (norm_Rx + norm_Ry + norm_Rc + norm_cg + norm_dg + norm_ug + norm_lg + norm_rg)

            # λ_hat (numeri freddi, perché S e norm_* sono detach)
            lambda_Rx_hat = S / (norm_Rx + self.eps)
            lambda_Ry_hat = S / (norm_Ry + self.eps)
            lambda_Rc_hat = S / (norm_Rc + self.eps)
            lambda_c_hat  = S / (norm_cg + self.eps)
            lambda_d_hat  = S / (norm_dg + self.eps)
            lambda_u_hat  = S / (norm_ug + self.eps)
            lambda_l_hat  = S / (norm_lg + self.eps)
            lambda_r_hat  = S / (norm_rg + self.eps)

            # EMA con detach a valle (comunque siamo in no_grad)
            lambda_Rx = self.alpha * w[0] + (1 - self.alpha) * lambda_Rx_hat
            lambda_Ry = self.alpha * w[1] + (1 - self.alpha) * lambda_Ry_hat
            lambda_Rc = self.alpha * w[2] + (1 - self.alpha) * lambda_Rc_hat
            lambda_c  = self.alpha * w[3] + (1 - self.alpha) * lambda_c_hat
            lambda_d  = self.alpha * w[4] + (1 - self.alpha) * lambda_d_hat 
            lambda_u  = self.alpha * w[5] + (1 - self.alpha) * lambda_u_hat 
            lambda_l  = self.alpha * w[6] + (1 - self.alpha) * lambda_l_hat
            lambda_r  = self.alpha * w[7]+ (1 - self.alpha) * lambda_r_hat 

            weights = torch.tensor([lambda_Rx, lambda_Ry, lambda_Rc,
                                lambda_c, lambda_d, lambda_u, lambda_l, lambda_r], device=self.device)

            print(f'Lambda weights: {[f"{weight:.3f}" for weight in weights.tolist()]}')

        return  total_loss, loss_PDE, loss_BC, weights      
    






