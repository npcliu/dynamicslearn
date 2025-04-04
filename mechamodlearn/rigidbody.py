# File: rigidbody.py

import abc
import torch

from mechamodlearn import nn, utils
from mechamodlearn.models import CholeskyMMNet, PotentialNet, GeneralizedForceNet, InvdynconsensusNet, NnmodelableDynNet, CholeskyMMNet1, nonlineardynNet, NnmodelableKinNet


class AbstractRigidBody:
    # def __init__(self, qdim):
    #     self.last_inv_dyn = torch.zeros((500, qdim), device='cuda:0')
    #     self.last_forward_dyn = torch.zeros((500, qdim), device='cuda:0')
        
    @property
    @abc.abstractmethod
    def thetamask(self):
        """Returns theta mask for configuration q.
        These should use utils.diffangles to compute differences
        """

    @abc.abstractmethod
    def mass_matrix(self, q, v, choose):
        """Return mass matrix for configuration q"""

    @abc.abstractmethod
    def potential(self, q, choose):
        """Return potential for configuration q"""

    @abc.abstractmethod
    def generalized_force(self, q, v, choose):
        """Return generalized force for configuration q, velocity v, external torque u"""

        
    def reset_buffer(self, q, u):
        self.last_inv_dyn_u = torch.zeros_like(u)
        self.last_forward_dyn_qddot = torch.zeros_like(q)

    def kinetic_energy(self, q, v, choose):
        mass_matrix = self.mass_matrix(q, v, choose)
        # TODO(jkg): Check if this works correctly for batched
        kenergy = 0.5 * (v.unsqueeze(1) @ (mass_matrix @ v.unsqueeze(2))).squeeze(2)
        return kenergy

    def lagrangian(self, q, v, choose):
        """ Returns the Lagrangian of a mechanical system
        """
        kenergy = self.kinetic_energy(q, v, choose)
        pot = self.potential(q, choose)
        lag = kenergy - pot
        return lag

    def hamiltonian(self, q, v, choose):
        """ Returns the Hamiltonian of a mechanical system
        """
        kenergy = self.kinetic_energy(q, v, choose)
        pot = self.potential(q, choose)
        ham = kenergy + pot
        return ham

    def corriolisforce(self, q, v, mass_matrix=None):
        """ Computes the corriolis matrix times v
        """
        with torch.enable_grad():

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

    def gradpotential(self, q, choose):
        """ Returns the conservative forces acting on the system
        """
        with torch.enable_grad():
            pot = self.potential(q, choose)
            gvec = torch.autograd.grad(torch.sum(pot), q, retain_graph=True, create_graph=True)[0]
        return gvec

    def solve_euler_lagrange(self, q, v, u, last_qddot, delta_t):
        """ Computes `qddot` (generalized acceleration) by solving
        the Euler-Lagrange equation (Eq 7 in the paper)
        \qddot = M^-1 (F - Cv - G)
        """
        with torch.enable_grad():
            with utils.temp_require_grad((q, v)):
                M = self.mass_matrix(q, v, 'forw')
                Cv = self.corriolisforce(q, v, M)
                G = self.gradpotential(q, 'forw')

        # F = torch.zeros_like(Cv)

        # if u is not None:
        #     F = self.generalized_force(q, v)
        F = self.generalized_force(q, v, 'forw')
        
        # Solve M \qddot = F - Cv - G
        # qddot = torch.gesv(F - Cv - G.unsqueeze(2), M)[0].squeeze(2)
        solve = torch.linalg.solve(M, u.unsqueeze(-1) + F - Cv - G.unsqueeze(2)).squeeze(2)
        delta_qddot = solve - self.last_forward_dyn_qddot
        qddot = last_qddot + delta_qddot * delta_t
        self.last_forward_dyn_qddot = solve.detach().clone()
        return qddot


class LearnedRigidBody(AbstractRigidBody, torch.nn.Module):

    def __init__(self, qdim: int, udim: int, thetamask: torch.tensor):
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

        # if mass_matrix is None:
        #     mass_matrix = CholeskyMMNet(qdim, hidden_sizes=[16, 32, 64])
        #     # mass_matrix = CholeskyMMNet1(qdim, hidden_sizes=[16, 32, 64])
        self._inv_mass_matrix = CholeskyMMNet(qdim, hidden_sizes=[24, 32, 64])

        # if potential is None:
        #     potential = PotentialNet(qdim, hidden_sizes=[16, 16])
        self._inv_potential = PotentialNet(qdim, hidden_sizes=[16, 16])

        # if generalized_force is None:
        #     # generalized_force = GeneralizedForceNet(qdim, udim, hidden_sizes)
        #     generalized_force = GeneralizedForceNet(qdim, hidden_sizes=[32, 16])
        self._inv_generalized_force = GeneralizedForceNet(qdim, hidden_sizes=[32, 16])
        
        self._forw_mass_matrix = CholeskyMMNet(qdim, self._inv_mass_matrix.embed, hidden_sizes=[24, 32, 64])
        self._forw_potential = PotentialNet(qdim, self._inv_potential.embed, hidden_sizes=[16, 16])
        self._forw_generalized_force = GeneralizedForceNet(qdim, hidden_sizes=[32, 16])
        
        self._unmodelable_kin = NnmodelableKinNet((qdim*3)+udim, hidden_sizes=[64, 32, 24], output_dim=qdim*2)
        
        # self._invdyn_consensus = InvdynconsensusNet((qdim*5), hidden_sizes=[64, 64, 32, 16], output_dim=qdim)
        # # udim+1:tau的维度加上delta_t的维度
        # self._unmodelable_dyn = NnmodelableDynNet((qdim*3)+udim+1, hidden_sizes=[64, 64, 32, 16], output_dim=qdim)
        # self._unmodelable_inv_dyn = InvdynconsensusNet((qdim*4), hidden_sizes=[64, 64, 32, 16], output_dim=qdim)
        
        # self._nonlinear_dyn = nonlineardynNet((qdim*6)+1, hidden_sizes=[64, 128, 64, 32, 16], output_dim=qdim)

    def mass_matrix(self, q, v, choose):
        if choose == 'inv':
            return self._inv_mass_matrix(q, v)
        elif choose == 'forw':
            return self._forw_mass_matrix(q, v)

    def potential(self, q, choose):
        if choose == 'inv':
            return self._inv_potential(q)
        elif choose == 'forw':
            return self._forw_potential(q)

    def generalized_force(self, q, v, choose):
        if choose == 'inv':
            return self._inv_generalized_force(q, v)
        elif choose == 'forw':
            return self._forw_generalized_force(q, v)
    

    # def invdyn_consensus(self, q, v, qddot, q_pre, v_pre):
    #     return self._invdyn_consensus(q, v, qddot, q_pre, v_pre)

    # def unmodelable_dyn(self, q, v, qddot, u, delta_t):
    #     return self._unmodelable_dyn(q, v, qddot, u, delta_t)
    
    # def unmodelable_inv_dyn(self, q, v, qddot, v_pre):
    #     return self._unmodelable_inv_dyn(q, v, qddot, v_pre)

    def delta_qv(self, q, v, qddot, u):
        out = self._unmodelable_kin(q, v, qddot, u)
        return out[:, :self._qdim].squeeze(2), out[:, self._qdim:].squeeze(2)

    def inv_dynamics(self, q, v, qddot, last_q, last_v, last_qddot, last_u, delta_t):
        # # 机器人逆动力学，计算力矩
        with torch.enable_grad():
            with utils.temp_require_grad((q, v)):
                # delta_M = self.mass_matrix(q)
                M = self.mass_matrix(q, v, 'inv')
                Cv = self.corriolisforce(q, v, M)
                G = self.gradpotential(q, 'inv')
                # self.M = M.clone()
                # self.Cv = Cv.clone()
                # self.G = G.clone()
        # F = torch.zeros_like(Cv)
        F = self.generalized_force(q, v, 'inv')
        # self.F = F.clone()
        # invdyn_cons = self.invdyn_consensus(q, v, qddot, q_pre, v_pre)
        # qddot_solve = qddot - self.unmodelable_inv_dyn(q, v, qddot, v_pre).squeeze(2)
        qddot_solve = qddot
        result = torch.bmm(M, qddot.unsqueeze(2))
        inv_dyn = (result + Cv + G.unsqueeze(2) - F).squeeze(2)
        delta_u_hat = inv_dyn - self.last_inv_dyn_u
        u_hat = last_u + delta_u_hat * delta_t
        self.last_inv_dyn_u = inv_dyn.detach().clone()
        # self.last_F = F.clone()
        
        # result = torch.bmm(delta_M, (qddot - last_qddot).unsqueeze(2))
        # delta_u_hat = (result + self._nonlinear_dyn(q, v, qddot, last_q, last_v, last_qddot, delta_t))
        # u_hat = last_u + delta_u_hat.squeeze(2) * delta_t
        return u_hat, qddot_solve
    @property
    def thetamask(self):
        return self._thetamask

    def forward(self, q, v, u, last_qddot, delta_t):
        qddot_solve = self.solve_euler_lagrange(q, v, u, last_qddot, delta_t)
        # qddot = self.unmodelable_dyn(q, v, qddot_solve, u, delta_t).squeeze(2) + qddot_solve
        qddot = qddot_solve
        return qddot, qddot_solve
