# ASD(FGHJK)L

`asdfghjkl` is integrated from release [v0.1a2](https://github.com/kazukiosawa/asdl/releases/tag/v0.1a2) of the [asdl library](https://github.com/kazukiosawa/asdl).
The name stands for **A**utomatic **S**econd-order **D**ifferentiation (for **F**isher, **G**radient covariance, **H**essian, **J**acobian, and **K**ernel) **L**ibrary.
Here, we particularly focus on the functionality relevent to Laplace approximations, that is, estimation of the curvature matrices.

### Basic metrics supported by a standard automatic differentiation libarary (ADL)
| metric | definition |
| --- | --- |
| neural network | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;f_\theta:\mathbb{R}^{M_{0}}\to\mathbb{R}^{C},\,\,\,\theta\in\mathbb{R}^{P}"/> |
| loss | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^N\ell(x_i,y_i,\theta)=\left\langle\ell(x_i,y_i,\theta)\right\rangle"/> |
| (averaged) gradient | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\bar{g}=\nabla\mathcal{L}(\theta)=\left\langle\frac{\partial}{\partial\theta}\ell(x_i,y_i,\theta)\right\rangle=\left\langle\mathbf{J}_{f,\theta}(x_i)^\top\frac{\partial}{\partial{f}}\ell(x_i,y_i,\theta)\right\rangle\in\mathbb{R}^P"/> |

### Advanced 1st/2nd-order metrics (FGHJK) supported by ASDL
| metric | definition |
| --- | --- |
| **F**isher information matrix | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\mathbf{F}=\left\langle\mathbb{E}_{p(k\|x_i)}\left[\frac{\partial}{\partial\theta}\ell(x_i,k,\theta)\frac{\partial}{\partial\theta}\ell(x_i,k,\theta)^\top\right]\right\rangle\in\mathbb{R}^{P\times{P}}" />  |
| **F**isher information matrix (MC estimation) | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\mathbf{F}_{n{\rm{mc}}}=\left\langle\frac{1}{n}\sum_{j=1}^n\frac{\partial}{\partial\theta}\ell(x_i,k^{(j)},\theta)\frac{\partial}{\partial\theta}\ell(x_i,k^{(j)},\theta)^\top\right\rangle\in\mathbb{R}^{P\times{P}},\,\,\,k^{(j)}\sim{p(k\|x)}" />  |
| empirical **F**isher | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\mathbf{F}_{\rm{emp}}=\left\langle\frac{\partial}{\partial\theta}\ell(x_i,y_i,\theta)\frac{\partial}{\partial\theta}\ell(x_i,y_i,\theta)^\top\right\rangle\in\mathbb{R}^{P\times{P}}" />  |
| **G**radient covariance | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\mathbf{C}=\left\langle\left(\frac{\partial}{\partial\theta}\ell(x_i,y_i,\theta)-\bar{g}\right)\left(\frac{\partial}{\partial\theta}\ell(x_i,y_i,\theta)-\bar{g}\right)^\top\right\rangle\in\mathbb{R}^{P\times{P}}" />  |
| **H**essian | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\mathbf{H}=\nabla^2\mathcal{L}(\theta)=\left\langle\frac{\partial^2}{\partial\theta\partial\theta^\top}\ell(x_i,y_i,\theta)\right\rangle\in\mathbb{R}^{P\times{P}}"/> |
| **J**acobian (per example) | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\mathbf{J}_{f,\theta}(x)=\frac{\partial}{\partial\theta}f_{\theta}(x)\in\mathbb{R}^{C\times{P}}"/> |
| **J**acobian | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\mathcal{J}=\left[\mathbf{J}_{f,\theta}(x_1)^\top,\dots,\mathbf{J}_{f,\theta}(x_N)^\top\right]^\top\in\mathbb{R}^{NC\times{P}}"/> |
| **K**ernel | <img src="https://latex.codecogs.com/png.latex?\dpi{130}&space;\mathcal{K}=\mathcal{JJ}^\top\in\mathbb{R}^{NC\times{NC}}"/> |

## Matrix approximations
<img src="https://user-images.githubusercontent.com/7961228/122673899-05c47d80-d1d3-11eb-8ece-74063a029666.png" width="800"/>
