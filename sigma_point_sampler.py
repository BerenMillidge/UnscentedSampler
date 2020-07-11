class SigmaPointSampler(nn.Module):
  def __init__(self, n,alpha,beta,kappa=None,use_sigma_weights=False,device="cpu"):
    super().__init__()
    self.n = n
    self.alpha = alpha
    self.beta = beta
    if kappa is None:
      self.kappa = 3-n
    else:
      self.kappa = kappa
    self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
    self.num_sigmas = (2*self.n) + 1
    self.c =  0.5 / (self.n + self.lambda_)
    self.use_sigma_weights = use_sigma_weights
    self.device=device

  def forward(self,f,mu,var,actions):
    ensemble_size, batch_size, state_dim = mu.size()
    action_dim = actions.size()[-1]
    x_sigmas = mu.unsqueeze(3).repeat(1,1,1,self.num_sigmas)
    sigma_add = torch.diag_embed(torch.sqrt((self.n + self.lambda_) *var))
    zero_add = torch.zeros_like(mu).unsqueeze(3)
    x_sigmas += torch.cat([zero_add,sigma_add, -sigma_add],dim=3)
    actions = actions.unsqueeze(3).repeat(1,1,1,self.num_sigmas) 
    x_sigmas = x_sigmas.view(ensemble_size, -1, state_dim)
    actions = actions.view(ensemble_size, -1, 1)
    y_sigmas = f(x_sigmas,actions)
    y_sigmas = y_sigmas.view(ensemble_size, batch_size, state_dim, self.num_sigmas)
    if self.use_sigma_weights:
      Wm = torch.full([self.num_sigmas,1],self.c).to(self.device)
      Wc = torch.full([self.num_sigmas,1],self.c).to(self.device)
      Wm[0] = self.lambda_ / (self.n+self.lambda_)
      Wc[0] = (self.lambda_ / (self.n+ self.lambda_)) + (1 - self.alpha**2 + self.beta)
      new_mu = torch.tensordot(y_sigmas, Wm, dims=([3],[0]))
      new_var = torch.tensordot((y_sigmas -torch.mean(y_sigmas,dim=1,keepdim=True))**2,Wc,dims=([3],[0]))
      return new_mu.squeeze(3), new_var.squeeze(3)
    else:
      mean_mu = torch.mean(y_sigmas,dim=3)
      mean_var = torch.var(y_sigmas, dim=3) 
      return mean_mu,mean_var
