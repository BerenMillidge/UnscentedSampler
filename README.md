# UnscentedSampler
Pytorch class implementing Unscented sigma point sampling to improve estimation of nonlinear dynamics. Given a nonlinear transition model, this class uses sigma point sampling to maintain an improved estimate of uncertainty propagated through the nonlinear dynamics. 

Based heavily on ideas from the Unscented Kalman Filter (UKF) -- https://onlinelibrary.wiley.com/doi/abs/10.1002/0471221546#page=234

This class provides an unscented filtering pytorch module which can be used in a modular plug-and-play module in deep neural networks for nonlinear dynamics estimation.
