# File: models.py
#
import torch

from mechamodlearn import nn, utils

GAIN=0.1
LN = False#True
class SharedMMVEmbed(torch.nn.Module):

    def __init__(self, qdim, hidden_sizes, activation='tanh'):
        self._qdim = qdim
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        super().__init__()
        # self._lnet = nn.LNMLP(qdim, hidden_sizes[:-1], hidden_sizes[-1], activation=activation, gain=GAIN, ln=LN)
        self._lnet = nn.ResMLP(qdim, hidden_sizes[:-1], hidden_sizes[-1], ln=LN)
        
        
        
    def forward(self, q):
        embed = self._lnet(q)
        return embed

class CholeskyMMNet(torch.nn.Module):

    def __init__(self, qdim, embed=None, hidden_sizes=None, bias=2):
        super().__init__()
        self._qdim = qdim
        self._bias = bias

        if embed is None:
            if hidden_sizes is None:
                raise ValueError("embed and hidden_sizes; both can't be None")
            embed = SharedMMVEmbed(qdim, hidden_sizes, activation='relu')

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], int(qdim * (qdim + 1) / 2))

    def forward(self, q):
        B = q.size(0)
        if self._qdim > 1:
            L_params = self.out(self.embed(q))
            L_diag = L_params[:, :self._qdim]
            L_diag += self._bias
            L_tril = L_params[:, self._qdim:]
            L = q.new_zeros(B, self._qdim, self._qdim)
            L = utils.bfill_lowertriangle(L, L_tril)
            L = utils.bfill_diagonal(L, L_diag)
            M = L @ L.transpose(-2, -1)

        else:
            M = self._pos_enforce((self.out(self.embed(q)) + self._bias).unsqueeze(1))

        return M


class PotentialNet(torch.nn.Module):

    def __init__(self, qdim, embed=None, hidden_sizes=None):
        self._qdim = qdim
        super().__init__()
        if embed is None:
            if hidden_sizes is None:
                raise ValueError("embed and hidden_sizes; both can't be None")

            embed = SharedMMVEmbed(qdim, hidden_sizes, activation='relu')

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, q):
        # print(self.embed._hidden_sizes)
        return self.out(self.embed(q))


class GeneralizedForceNet(torch.nn.Module):

    def __init__(self, qdim, hidden_sizes):
        self._qdim = qdim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        # self._net = nn.LNMLP(self._qdim * 2, hidden_sizes, qdim, activation='elu', gain=GAIN, ln=LN)
        self._net = nn.ResMLP(self._qdim * 2, hidden_sizes, qdim, ln=LN)
        
    def forward(self, q, v):
        B = q.size(0)
        x = torch.cat([q, v], dim=-1)
        F = self._net(x)
        F = F.unsqueeze(2)
        assert F.shape == (B, self._qdim, 1), F.shape
        return F

class InvdynconsensusNet(torch.nn.Module):
# midpoint和rk4积分方法都要计算好几次正动力学，那正动力学的网络梯度不是
# 直接q,v到tau产生的，而是好几个中间值产生的，而泥动力学不产生中间值，所以加一个小网络
# 保持正逆动力学一致性
    def __init__(self, input_dim, hidden_sizes, output_dim):
        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._output_dim = output_dim
        super().__init__()
        # self._net = nn.LNMLP(self._input_dim, hidden_sizes, output_dim, activation='elu', gain=GAIN, ln=LN)
        self._net = nn.ResMLP(self._input_dim, hidden_sizes, output_dim, ln=LN)
        
    def forward(self, q, v, qddot, v_pre):
        B = q.size(0)
        x = torch.cat([q, v, qddot, v_pre], dim=-1)
        F = self._net(x)
        F = F.unsqueeze(2)
        assert F.shape == (B, self._output_dim, 1), F.shape
        return F

class NnmodelableDynNet(torch.nn.Module):
# midpoint和rk4积分方法都要计算好几次正动力学，那正动力学的网络梯度不是
# 直接q,v到tau产生的，而是好几个中间值产生的，而泥动力学不产生中间值，所以加一个小网络
# 保持正逆动力学一致性
    def __init__(self, input_dim, hidden_sizes, output_dim):
        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._output_dim = output_dim
        super().__init__()
        # self._net = nn.LNMLP(input_dim, hidden_sizes, output_dim, activation='elu', gain=GAIN, ln=LN)
        self._net = nn.ResMLP(input_dim, hidden_sizes, output_dim, ln=LN)
        
    def forward(self, q, v, qddot, u, delta_t):
        B = q.size(0)
        x = torch.cat([q, v, qddot, u, delta_t], dim=-1)
        F = self._net(x)
        F = F.unsqueeze(2)
        assert F.shape == (B, self._output_dim, 1), F.shape
        return F
    
class ControlAffineForceNet(torch.nn.Module):

    def __init__(self, qdim, udim, hidden_sizes):
        self._qdim = qdim
        self._udim = udim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        # self._net = nn.LNMLP(self._qdim, hidden_sizes, qdim * udim)
        self._net = nn.ResMLP(self._qdim, hidden_sizes, qdim * udim)

    def forward(self, q, v, u):
        B = q.size(0)
        Bmat = self._net(q).view(B, self._qdim, self._udim)
        F = Bmat @ u.unsqueeze(2)
        assert F.shape == (B, self._qdim, 1), F.shape
        return F


class ControlAffineLinearForce(torch.nn.Module):

    def __init__(self, B):
        """
        B needs to be shaped (1, qdim, qdim) usually diagonal
        """
        super().__init__()
        if not isinstance(B, torch.nn.Parameter):
            B = torch.nn.Parameter(B)

        self._B = B

    def forward(self, q, v, u):
        N = q.size(0)
        assert u.size(0) == N
        assert self._B.shape == (1, q.size(1), q.size(1)), self._B.shape
        B = self._B
        F = B @ u.unsqueeze(2)
        assert F.shape == (N, q.size(1), 1), F.shape
        return F


class ViscousJointDampingForce(torch.nn.Module):

    def __init__(self, eta):
        """
        eta needs to be shaped (1, qdim)
        """
        super().__init__()
        if not isinstance(eta, torch.nn.Parameter):
            eta = torch.nn.Parameter(eta)

        self._eta = eta

    def forward(self, q, v, u):
        N = q.size(0)
        assert self._eta.size(1) == v.size(1)
        F = (self._eta * v).unsqueeze(2)
        assert F.shape == (N, q.size(1), 1), F.shape
        return F


class GeneralizedForces(torch.nn.Module):

    def __init__(self, forces):
        super().__init__()
        self.forces = torch.nn.ModuleList(forces)

    def forward(self, q, v, u):
        F = torch.zeros(q.size(0), q.size(1), 1, device='cuda:0')
        for f in self.forces:
            F += f(q, v, u)

        return F
