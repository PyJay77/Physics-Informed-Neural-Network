# PINNs for 2D Incompressible Navier–Stokes (steady & unsteady) - Flow past a cylinder

```
2D_NavierStokes_steady/    # PINN for the steady solution (u, v, p)
2D_NavierStokes_unsteady/  # PINN for the unsteady problem, initialized from the steady state
```

This repository contains a **data-free Physics-Informed Neural Network (PINN)** implementation in **PyTorch** for the 2D incompressible Navier–Stokes equations around a circular cylinder. The inflow is **parabolic** with maximum speed $U_{\max}=0.05$ m/s. We first solve a **steady** problem and then **use that solution as the initial condition** for the **unsteady** problem.

The implementation follows the training pipeline described in *An expert’s guide to training Physics-Informed Neural Networks* (Wang et al., 2023), implementing it with **PyTorch**.

---

## 1) Physical problem & Boundary conditions
The physical problem we simulate is a 2d flow around a cylinder. The dimensions of the domain are the same used in the paper named before, in particular:

$x \in [0,2.2]$ m  
$y \in [0,0.41]$ m  
$t \in [0,10]$ s

The parameters of the cylinder are:

$x_c = 0.4$ m   
$y_c = 0.2$ m  
$r = 0.05$ m  


The boundary conditions are:

### Inlet (parabolic profile)
At the inlet, a parabolic velocity profile is imposed:

$$
u(y) = \frac{4 \cdot U_{\max}}{H^2} \cdot y \cdot (H - y), \quad v(y) = 0, \quad y \in [0,H]
$$

### Walls & Cylinder (no-slip)
No-slip condition is imposed on all walls and the cylinder:

$$
u = 0, \quad v = 0
$$

### Outlet (zero-stress)
The stress tensor for an incompressible fluid is:

$$
\sigma = -p I + \mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right)
$$

The zero-stress condition at the outlet requires the normal component of the stress to vanish:

$$
\sigma \cdot \mathbf{n} = 0
$$

For a 2D outlet with normal $\mathbf{n} = (1,0)$ (flow along $x$), this gives:

$$
(\sigma \cdot \mathbf{n})_x = \sigma_{xx} n_x + \sigma_{xy} n_y = \sigma_{xx} = -p + 2 \mu \frac{\partial u}{\partial x} = 0
$$

$$
(\sigma \cdot \mathbf{n})_y = \sigma_{yx} n_x + \sigma_{yy} n_y = \sigma_{yx} = \mu \left( \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y} \right) = 0
$$

---

## 2) Non-dimensionalization of 2D Incompressible Navier–Stokes

We start from the **dimensional 2D incompressible Navier–Stokes equations**:

$$
\rho \left( \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} \right) 
= - \frac{\partial p}{\partial x} + \mu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

$$
\rho \left( \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} \right) 
= - \frac{\partial p}{\partial y} + \mu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

where $u, v$ are the dimensional velocities [m/s], $p$ is pressure [Pa], $\rho$ is density, and $\mu$ is dynamic viscosity.


### 1) Reference scales

We define characteristic scales:

- **Length:** $D$ (cylinder diameter)  
- **Velocity:** $U_{\max}$ (maximum inlet velocity)  
- **Time:** $T_{\mathrm{ref}} = D / U_{\max}$  
- **Pressure:** $P_{\mathrm{ref}} = \rho U_{\max}^2$  


### 2) Dimensionless variables

Introduce **dimensionless variables**:

$$
x^\ast = \frac{x}{D}, \quad y^\ast = \frac{y}{D}, \quad
t^\ast = t \frac{U_{\max}}{D}, \quad
U^\ast = \frac{u}{U_{\max}}, \quad
V^\ast = \frac{v}{U_{\max}}, \quad
P^\ast = \frac{p}{\rho U_{\max}^2}.
$$

Derivatives transform as:

$$
\frac{\partial}{\partial x} = \frac{1}{D} \frac{\partial}{\partial x^\ast}, \quad
\frac{\partial}{\partial y} = \frac{1}{D} \frac{\partial}{\partial y^\ast}, \quad
\frac{\partial}{\partial t} = \frac{U_{\max}}{D} \frac{\partial}{\partial t^\ast}, \quad
\frac{\partial^2}{\partial x^2} = \frac{1}{D^2} \frac{\partial^2}{\partial {x^\ast}^2}, \quad
\frac{\partial^2}{\partial y^2} = \frac{1}{D^2} \frac{\partial^2}{\partial {y^\ast}^2}.
$$


### 3) Substitution into Navier–Stokes

Substituting into the dimensional equations:

$$
\rho \frac{U_{\max}^2}{D} \left( \frac{\partial U^\ast}{\partial t^\ast} + U^\ast \frac{\partial U^\ast}{\partial x^\ast} + V^\ast \frac{\partial U^\ast}{\partial y^\ast} \right) 
= - \rho U_{\max}^2 \frac{1}{D} \frac{\partial P^\ast}{\partial x^\ast} + \mu \frac{U_{\max}}{D^2} \left( \frac{\partial^2 U^\ast}{\partial {x^\ast}^2} + \frac{\partial^2 U^\ast}{\partial {y^\ast}^2} \right)
$$

$$
\rho \frac{U_{\max}^2}{D} \left( \frac{\partial V^\ast}{\partial t^\ast} + U^\ast \frac{\partial V^\ast}{\partial x^\ast} + V^\ast \frac{\partial V^\ast}{\partial y^\ast} \right) 
= - \rho U_{\max}^2 \frac{1}{D} \frac{\partial P^\ast}{\partial y^\ast} + \mu \frac{U_{\max}}{D^2} \left( \frac{\partial^2 V^\ast}{\partial {x^\ast}^2} + \frac{\partial^2 V^\ast}{\partial {y^\ast}^2} \right)
$$

Divide through by $\rho U_{\max}^2 / D$:

$$
\frac{\partial U^\ast}{\partial t^\ast} + U^\ast \frac{\partial U^\ast}{\partial x^\ast} + V^\ast \frac{\partial U^\ast}{\partial y^\ast} 
= - \frac{\partial P^\ast}{\partial x^\ast} + \frac{\mu}{\rho U_{\max} D} \left( \frac{\partial^2 U^\ast}{\partial {x^\ast}^2} + \frac{\partial^2 U^\ast}{\partial {y^\ast}^2} \right)
$$

$$
\frac{\partial V^\ast}{\partial t^\ast} + U^\ast \frac{\partial V^\ast}{\partial x^\ast} + V^\ast \frac{\partial V^\ast}{\partial y^\ast} 
= - \frac{\partial P^\ast}{\partial y^\ast} + \frac{\mu}{\rho U_{\max} D} \left( \frac{\partial^2 V^\ast}{\partial {x^\ast}^2} + \frac{\partial^2 V^\ast}{\partial {y^\ast}^2} \right)
$$

$$
\frac{\partial U^\ast}{\partial x^\ast} + \frac{\partial V^\ast}{\partial y^\ast} = 0
$$


### 4) Introducing Reynolds number

Define the **Reynolds number**:

$$
\mathrm{Re} = \frac{\rho U_{\max} D}{\mu} = \frac{U_{\max} D}{\nu}, \quad \nu = \frac{\mu}{\rho}.
$$

The **dimensionless 2D incompressible Navier–Stokes equations** are then:

$$
\frac{\partial U^\ast}{\partial t^\ast} + U^\ast \frac{\partial U^\ast}{\partial x^\ast} + V^\ast \frac{\partial U^\ast}{\partial y^\ast} 
= - \frac{\partial P^\ast}{\partial x^\ast} + \frac{1}{\mathrm{Re}} \left( \frac{\partial^2 U^\ast}{\partial {x^\ast}^2} + \frac{\partial^2 U^\ast}{\partial {y^\ast}^2} \right)
$$

$$
\frac{\partial V^\ast}{\partial t^\ast} + U^\ast \frac{\partial V^\ast}{\partial x^\ast} + V^\ast \frac{\partial V^\ast}{\partial y^\ast} 
= - \frac{\partial P^\ast}{\partial y^\ast} + \frac{1}{\mathrm{Re}} \left( \frac{\partial^2 V^\ast}{\partial {x^\ast}^2} + \frac{\partial^2 V^\ast}{\partial {y^\ast}^2} \right)
$$

$$
\frac{\partial U^\ast}{\partial x^\ast} + \frac{\partial V^\ast}{\partial y^\ast} = 0
$$

> These are the final **dimensionless Navier–Stokes equations** used for PINN training.


---

## 3) PINN formulation

We approximate the unknown fields with a neural network

$$
f_\theta:(t,x,y)\mapsto(u,v,p)
$$

and minimize the **weighted composite loss**

$$
\mathcal{L}(\theta)=
\lambda_{\mathrm{ic}}\mathcal{L}_{\mathrm{ic}}+
\lambda_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}+
\lambda_{\mathrm{r}}\mathcal{L}_{\mathrm{r}}
$$


### 3.1 PDE residuals

**Unsteady (dimensionless):**

$$
\begin{aligned}
r_u &= u_t + u \cdot u_x + v \cdot u_y + p_x - \frac{1}{\mathrm{Re}}(u_{xx}+u_{yy}),\\
r_v &= v_t + u \cdot v_x + v \cdot v_y + p_y - \frac{1}{\mathrm{Re}}(v_{xx}+v_{yy}),\\
r_c &= u_x + v_y
\end{aligned}
$$

**Steady:** same expressions without \(u_t, v_t\).

**Residual loss over interior collocation points:**

$$
\mathcal{L}_{\mathrm{r}}=\frac{1}{N_r}\sum_{i=1}^{N_r}(r_u^2+r_v^2+r_c^2)
$$


### 3.2 Boundary & initial condition losses

**Dirichlet boundaries** (inlet, walls, cylinder):

$$
\mathcal{L}_{\mathrm{bc}}=\frac{1}{N_{\mathrm{bc}}}\sum\|(u,v)-(u,v)_{\text{target}}\|^2
$$

**Weak-Neumann outlet:** gradient constraints on velocity + a pressure gauge constraint.

**IC (unsteady):**

$$
\mathcal{L}_{\mathrm{ic}}=\frac{1}{N_{\mathrm{ic}}}\sum\|(u,v)-(u,v)_{\text{steady}}\|^2
$$


### 3.3 Causal weighting (time-dependent problems)

Split \([0,T]\) into \(M\) ordered segments and reweight the residual loss:

$$
\mathcal{L}_{\mathrm{r}}=\frac{1}{M}\sum_{i=1}^{M} w_i\mathcal{L}_{\mathrm{r}}^{(i)}(\theta)
$$

Update the causal weights each iteration:

$$
w_1=1,\qquad
w_i=\exp\left(-\varepsilon\sum_{k=1}^{i-1}\mathcal{L}_{\mathrm{r}}^{(k)}(\theta)\right),\quad i=2,\dots,M
$$


### 3.4 Adaptive loss balancing (global weights \(\lambda_\bullet\))

Every \(f\) iterations, compute provisional equalizing weights via gradient norms:

$$
\begin{aligned}
\hat\lambda_{\mathrm{ic}} &= 
\frac{\|\nabla_\theta \mathcal{L}_{\mathrm{ic}}\|+\|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\|+\|\nabla_\theta \mathcal{L}_{\mathrm{r}}\|}
{\|\nabla_\theta \mathcal{L}_{\mathrm{ic}}\|},\\
\hat\lambda_{\mathrm{bc}} &= 
\frac{\|\nabla_\theta \mathcal{L}_{\mathrm{ic}}\|+\|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\|+\|\nabla_\theta \mathcal{L}_{\mathrm{r}}\|}
{\|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\|},\\
\hat\lambda_{\mathrm{r}} &= 
\frac{\|\nabla_\theta \mathcal{L}_{\mathrm{ic}}\|+\|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\|+\|\nabla_\theta \mathcal{L}_{\mathrm{r}}\|}
{\|\nabla_\theta \mathcal{L}_{\mathrm{r}}\|}
\end{aligned}
$$

These ensure equal weighted gradient norms:

$$
\|\hat\lambda_{\mathrm{ic}}\nabla_\theta \mathcal{L}_{\mathrm{ic}}\|=
\|\hat\lambda_{\mathrm{bc}}\nabla_\theta \mathcal{L}_{\mathrm{bc}}\|=
\|\hat\lambda_{\mathrm{r}}\nabla_\theta \mathcal{L}_{\mathrm{r}}\|=
\|\nabla_\theta \mathcal{L}_{\mathrm{ic}}\|+
\|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\|+
\|\nabla_\theta \mathcal{L}_{\mathrm{r}}\|
$$

Then update the actual weights by EMA (no gradients through \(\lambda\)):

$$
\lambda^{\text{new}}=\alpha\,\lambda^{\text{old}}+(1-\alpha)\\hat\lambda
$$

### 3.5 Final objective and parameter update

Combining everything:

$$
\mathcal{L}(\theta)=
\lambda_{\mathrm{ic}}\mathcal{L}_{\mathrm{ic}}+
\lambda_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}+
\lambda_{\mathrm{r}}\left[\frac{1}{M}\sum_{i=1}^{M}w_i\,\mathcal{L}_{\mathrm{r}}^{(i)}(\theta)\right]
$$

Update network parameters with gradient descent / Adam:

$$
\theta_{n+1}=\theta_n-\eta\,\nabla_\theta \mathcal{L}(\theta_n)
$$

---

## 4) Network architecture

### Fourier features
The spatial–temporal inputs are first mapped to a higher-dimensional space using Fourier features:

$$
\gamma(\mathbf{x}) = [ \cos(B \mathbf{x}) \, \sin(B \mathbf{x}) ] \quad
B_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

where $B$ is a random Gaussian matrix with variance $\sigma^2$, controlling the frequency scale.

---

### Random Weight Factorization (RWF)
Each weight matrix is factorized as:

$$
W^{(\ell)} = \mathrm{diag}\left( e^{s^{(\ell)}} \right) \cdot V^{(\ell)}
$$

where $s^{(\ell)}$ are learnable log-scale parameters and $V^{(\ell)}$ are normalized base weights.  
This decomposition separates **magnitude** and **direction**, improving stability during training.

---

### Modified MLP layer
Each hidden layer combines two nonlinear transformations through a learned gate:

$$
U = \sigma(W_1 x + b_1), \quad V = \sigma(W_2 x + b_2)
$$

$$
g^{(\ell)} = \sigma(f_\theta^{(\ell)}) \odot U + \big(1 - \sigma(f_\theta^{(\ell)})\big) \odot V
$$

where \( \odot \) denotes element-wise multiplication and \( \sigma(\cdot) \) is the activation function (e.g., t


---

## 5) Repository layout





```
2D_NavierStokes_steady/
  ├── models                                        # Folder with the already trained neural network
  ├── NS_steady_training.ipynb                      # Jupyter notebook with the training loop
  ├── Results_check                                 # Jupyter notebook with the plots of the results obtained with the NN
  ├── loss_function_steady_optimized.py.py          # Python code containing the function for the loss_function
  ├── model_optimized.py                            # Python code containing the function of the neural network
  └──  points_generator_optimized.py                # Python code containing the function that generates the points to be given as input to the NN

2D_NavierStokes_unsteady/
  ├── models                                        # Folder with the already trained neural network with the steady state case
  ├── NS_unsteady_training.ipynb                    # Jupyter notebook with the training loop (NOT RUN YET)
  ├── loss_function_unsteady_optimized.py.py        # Python code containing the function for the loss_function with casualty weights algorith
  ├── model_unsteady_optimized.py                   # Python code containing the function of the neural network
  └── points_generator_optimized.py                 # Python code containing the function that generates the points to be given as input to the NN
```



---

## 6) Citation  

> S. Wang, S. Sankaran, H. Wang, P. Perdikaris (2023). *An expert’s guide to training Physics-Informed Neural Networks.* arXiv:2308.08468


