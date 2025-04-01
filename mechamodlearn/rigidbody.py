# File: rigidbody.py

import abc
import torch

from mechamodlearn import nn, utils
from mechamodlearn.models import CholeskyMMNet, PotentialNet, GeneralizedForceNet, InvdynconsensusNet, NnmodelableDynNet


class AbstractRigidBody:

    @property
    @abc.abstractmethod
    def thetamask(self):
        """Returns theta mask for configuration q.
        These should use utils.diffangles to compute differences
        """

    @abc.abstractmethod
    def mass_matrix(self, q):
        """Return mass matrix for configuration q"""

    @abc.abstractmethod
    def potential(self, q):
        """Return potential for configuration q"""

    @abc.abstractmethod
    def generalized_force(self, q, v):
        """Return generalized force for configuration q, velocity v, external torque u"""

    def kinetic_energy(self, q, v):
        mass_matrix = self.mass_matrix(q)
        # TODO(jkg): Check if this works correctly for batched
        kenergy = 0.5 * (v.unsqueeze(1) @ (mass_matrix @ v.unsqueeze(2))).squeeze(2)
        return kenergy

    def lagrangian(self, q, v):
        """ Returns the Lagrangian of a mechanical system
        """
        kenergy = self.kinetic_energy(q, v)
        pot = self.potential(q)
        lag = kenergy - pot
        return lag

    def hamiltonian(self, q, v):
        """ Returns the Hamiltonian of a mechanical system
        """
        kenergy = self.kinetic_energy(q, v)
        pot = self.potential(q)
        ham = kenergy + pot
        return ham

    def corriolisforce(self, q, v, mass_matrix=None):
        """ Computes the corriolis matrix times v
        """
        with torch.enable_grad():
            if mass_matrix is None:
                mass_matrix = self.mass_matrix(q)

            Mv = mass_matrix @ v.unsqueeze(2)

            KE = 0.5 * v.unsqueeze(1) @ Mv

            Cv_KE = torch.autograd.grad(KE.sum(), q, retain_graph=True, create_graph=True)[0]

            gMv = torch.stack([
                torch.autograd.grad(Mv[:, i].sum(), q, retain_graph=True, create_graph=True)[0]
                for i in range(q.size(1))
            ], dim=1)

            Cv = gMv @ v.unsqueeze(2) - Cv_KE.unsqueeze(2)

        return Cv

    def corriolis(self, q, v, mass_matrix=None):
        """ Computes the corriolis matrix
        """
        with torch.enable_grad():
            if mass_matrix is None:
                mass_matrix = self.mass_matrix(q)

            qdim = q.size(1)
            B = mass_matrix.size(0)

            mass_matrix = mass_matrix.reshape(-1, qdim, qdim)

            # TODO vectorize
            rows = []

            for i in range(qdim):
                cols = []
                for j in range(qdim):
                    qgrad = torch.autograd.grad(
                        torch.sum(mass_matrix[:, i, j]), q, retain_graph=True, create_graph=True)[0]
                    cols.append(qgrad)

                rows.append(torch.stack(cols, dim=1))

            dMijk = torch.stack(rows, dim=1)

        corriolis = 0.5 * ((dMijk + dMijk.transpose(2, 3) - dMijk.transpose(1, 3)
                           ) @ v.reshape(B, 1, qdim, 1)).squeeze(3)
        return corriolis

    def gradpotential(self, q):
        """ Returns the conservative forces acting on the system
        """
        with torch.enable_grad():
            pot = self.potential(q)
            gvec = torch.autograd.grad(torch.sum(pot), q, retain_graph=True, create_graph=True)[0]
        return gvec

    def solve_euler_lagrange(self, q, v, u=None):
        """ Computes `qddot` (generalized acceleration) by solving
        the Euler-Lagrange equation (Eq 7 in the paper)
        \qddot = M^-1 (F - Cv - G)
        """
        with torch.enable_grad():
            with utils.temp_require_grad((q, v)):
                M = self.mass_matrix(q)
                Cv = self.corriolisforce(q, v, M)
                G = self.gradpotential(q)
                self.M = M.clone()
                self.Cv = Cv.clone()
                self.G = G.clone()

        # F = torch.zeros_like(Cv)

        # if u is not None:
        #     F = self.generalized_force(q, v)
        F = self.generalized_force(q, v)
        self.F = F.clone()
        # Solve M \qddot = F - Cv - G
        # qddot = torch.gesv(F - Cv - G.unsqueeze(2), M)[0].squeeze(2)
        qddot = torch.linalg.solve(M, u.unsqueeze(-1) + F - Cv - G.unsqueeze(2)).squeeze(2)
        return qddot


class LearnedRigidBody(AbstractRigidBody, torch.nn.Module):

    def __init__(self, qdim: int, udim: int, thetamask: torch.tensor, mass_matrix=None,
                 potential=None, generalized_force=None, hidden_sizes=None):
        """

        Arguments:
        - `qdim`:
        - `udim`: [int]
        - `thetamask`: [torch.Tensor (1, qdim)] 1 if angle, 0 otherwise
        - `mass_matrix`: [torch.nn.Module]
        - `potential`: [torch.nn.Module]
        - `generalized_force`: [torch.nn.Module]
        - hidden_sizes: [list]
        """
        self._qdim = qdim
        self._udim = udim

        self._thetamask = thetamask

        super().__init__()

        if mass_matrix is None:
            mass_matrix = CholeskyMMNet(qdim, hidden_sizes=[16, 32, 64, 64])
            # mass_matrix = CholeskyMMNet(qdim, hidden_sizes=hidden_sizes)
        self._mass_matrix = mass_matrix

        if potential is None:
            potential = PotentialNet(qdim, hidden_sizes=[16, 32, 32, 16])
            # potential = PotentialNet(qdim, hidden_sizes=hidden_sizes)
        self._potential = potential

        if generalized_force is None:
            # generalized_force = GeneralizedForceNet(qdim, udim, hidden_sizes)
            generalized_force = GeneralizedForceNet(qdim, hidden_sizes=[32, 64, 64, 16])

        self._generalized_force = generalized_force
        self._invdyn_consensus = InvdynconsensusNet((qdim*4), hidden_sizes=[64, 64, 32, 16], output_dim=qdim)
        # udim+1:tau的维度加上delta_t的维度
        self._unmodelable_dyn = NnmodelableDynNet((qdim*3)+udim+1, hidden_sizes=[64, 64, 32, 16], output_dim=qdim)
        self._unmodelable_inv_dyn = InvdynconsensusNet((qdim*4), hidden_sizes=[64, 64, 32, 16], output_dim=qdim)
        self.M = torch.ones((500, self._qdim, self._qdim), device='cuda:0')
        self.Cv = torch.ones((500, self._qdim, 1), device='cuda:0')
        self.G = torch.ones((500, self._qdim), device='cuda:0')
        self.F = torch.ones((500, self._qdim, 1), device='cuda:0')
    def mass_matrix(self, q):
        return self._mass_matrix(q)

    def potential(self, q):
        return self._potential(q)

    def generalized_force(self, q, v):
        return self._generalized_force(q, v)
    
    def invdyn_consensus(self, q, v, qddot, v_pre):
        return self._invdyn_consensus(q, v, qddot, v_pre)

    def unmodelable_dyn(self, q, v, qddot, u, delta_t):
        return self._unmodelable_dyn(q, v, qddot, u, delta_t)
    
    def unmodelable_inv_dyn(self, q, v, qddot, v_pre):
        return self._unmodelable_inv_dyn(q, v, qddot, v_pre)
    
    
    def inv_dynamics(self, q, v, qddot, v_pre, delta_t):
        # 机器人逆动力学，计算力矩
        # F = torch.zeros_like(Cv)
        # F = self.generalized_force(q, v)
        invdyn_cons = self.invdyn_consensus(q, v, qddot, v_pre)
        qddot_solve = qddot - self.unmodelable_inv_dyn(q, v, qddot, v_pre).squeeze(2)
        # qddot_solve = qddot
        result = torch.bmm(self.M, qddot_solve.unsqueeze(2))
        u_hat = (result + self.Cv + self.G.unsqueeze(2) - self.F + invdyn_cons).squeeze(2)
        return u_hat, qddot_solve
    @property
    def thetamask(self):
        return self._thetamask

    def forward(self, q, v, u, delta_t):
        qddot_solve = self.solve_euler_lagrange(q, v, u)
        qddot = self.unmodelable_dyn(q, v, qddot_solve, u, delta_t).squeeze(2) + qddot_solve
        # qddot = qddot_solve
        return qddot, qddot_solve
