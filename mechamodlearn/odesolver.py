#
# File: odesolver.py
#

import abc
import torch
import numpy as np


class FixedGridODESolver(metaclass=abc.ABCMeta):

    def __init__(self, func, y0, grid_constructor=None, transforms=None):
        self.func = func
        self.y0 = y0

        if grid_constructor is None:
            grid_constructor = lambda f, y0, t: t

        self.grid_constructor = grid_constructor
        if transforms is None:
            transforms = [lambda x: x for _ in range(len(y0))]

        self.transforms = transforms

    @property
    @abc.abstractmethod
    def order(self):
        """Returns the integration order"""

    @abc.abstractmethod
    def step_func(self, func, t, dt, y, u):
        """Step once through"""

    def integrate(self, t, u=None):
        """
        Arguments:
        - `t`: timesteps to integrate over
        - `u` [list/torch.Tensor]: control inputs list for the time period
        """
        # _assert_increasing(t)
        if u is None:
            u = [None] * len(t)
        # print(u.shape)
        # print(self.y0)
        t = t.type_as(self.y0[0])
        time_grid = self.grid_constructor(self.func, self.y0, t) 
        # assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(self.y0[0])
        # time_grid = time_grid * 1000

        solution = [self.y0]
        u_hat_tuple = []
        u_tuple = []
        qddot_sol_forward_dyn_tuple = []
        qddot_sol_inv_dyn_tuple = []
        # print(u[0].shape)

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            delta_t = t1 - t0
            # delta_t = 0.001 * torch.ones_like(t1)
            # print(delta_t[100:110, 0])
            dy, qddot_sol_forward_dyn = self.step_func(self.func, t0, delta_t, y0, u=u[j - 1])
            y1 = tuple(trans(y0_ + dy_) for y0_, dy_, trans in zip(y0, dy, self.transforms))
            # print(len(t))
            # print(dy)
            # print(y1)
            # while j < len(t) and t1 >= t[j]:
            while j < len(t) and t1[0, 0] >= t[j][0, 0]:
                y_ = self._linear_interp(t0, t1, y0, y1, t[j])
                (q, v) = y0
                (q_pre, v_pre) = y1
                (_, qddotdt) = dy
                qddot = qddotdt / delta_t
                u_hat, qddot_sol_inv_dyn = self.func.diffeq.inv_dynamics(q, v, qddot, v_pre, delta_t)
                # u_hat, qddot_sol_inv_dyn = self.func.diffeq.inv_dynamics(q, v, qddot_sol_forward_dyn, q_pre, v_pre)
                solution.append(y_)
                u_hat_tuple.append(u_hat)
                u_tuple.append(u[j - 1])
                qddot_sol_forward_dyn_tuple.append(qddot_sol_forward_dyn)
                qddot_sol_inv_dyn_tuple.append(qddot_sol_inv_dyn)
                j += 1
            y0 = y1
        #     print(len(solution[0]), len(solution[0][0]), len(solution[0][0][0]))
        # print(len(solution[0]), len(solution[0][0]), len(solution[0][0][0]))
        # print(solution)
        u_hat_tensor = torch.stack(u_hat_tuple, dim=0)  
        u_tensor = torch.stack(u_tuple, dim=0)  
        qddot_sol_forward_dyn_tensor = torch.stack(qddot_sol_forward_dyn_tuple, dim=0)  
        qddot_sol_inv_dyn_tensor = torch.stack(qddot_sol_inv_dyn_tuple, dim=0)  
        # print(qddot_sol_forward_dyn_tensor[:, 0, 0])
        # print(qddot_sol_inv_dyn_tensor[:, 0, 0])
        # print(u_tensor[2, 2, :])
        # print(u[2, 2, :])
        return tuple(map(torch.stack, tuple(zip(*solution)))), u_hat_tensor, qddot_sol_forward_dyn_tensor, qddot_sol_inv_dyn_tensor, u_tensor

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t[0,0] == t0[0,0]:
        # if t == t0:
            return y0
        if t[0,0] == t1[0,0]:
        # if t == t1:
            return y1
        t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))
        # slope = tuple((y1_ - y0_) / (t1 - t0 + 8e-5) for y0_, y1_, in zip(y0, y1))
        # return tuple(y0_ + slope_ * (t - t0 + 8e-5) for y0_, slope_ in zip(y0, slope))


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y, u):
        k1, qddot_sol = func(dt, y, u=u)
        return tuple(dt * f_ for f_ in k1), qddot_sol

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y, u):
        k1, qddot_sol1 = func(dt / 2, y, u=u)
        y_mid = tuple(
            trans(y_ + f_ * dt / 2) for y_, f_, trans in zip(y, k1, self.transforms))
        k2, qddot_sol2 = func(dt / 2, y_mid, u=u)
        qddot_sol = (qddot_sol1 + qddot_sol2) / 2
        return tuple(dt * f_ for f_ in k2), qddot_sol

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y, u):
        return rk4_alt_step_func(func, t, dt, y, u=u)

    @property
    def order(self):
        return 4


def rk4_alt_step_func(func, t, dt, y, k1=None, u=None):
    """Smaller error with slightly more compute."""
    if k1 is None:
        k1 = func(t, y, u=u)
    # print(tuple(y_ + dt * k1_ / 3 for y_, k1_ in zip(y, k1)))
    k2 = func(t + dt / 3, tuple(y_ + dt * k1_ / 3 for y_, k1_ in zip(y, k1)), u=u)
    k3 = func(t + dt * 2 / 3,
              tuple(y_ + dt * (k1_ / -3 + k2_) for y_, k1_, k2_ in zip(y, k1, k2)), u=u)
    k4 = func(t + dt,
              tuple(y_ + dt * (k1_ - k2_ + k3_) for y_, k1_, k2_, k3_ in zip(y, k1, k2, k3)), u=u)
    # TODO:u_hat应该是输入的最终加速度也就是下面的元祖，所以rk4方法求逆动力学有问题
    return tuple((k1_ + 3 * k2_ + 3 * k3_ + k4_) * (dt / 8)
                 for k1_, k2_, k3_, k4_ in zip(k1, k2, k3, k4))


def odeint(func, y0, t, method=None, transforms=None, **kwargs):
    """Integrates `func` with initial conditions `y0` at points specified by `t`
    Arguments:
    - `func` : function to integrate: ydot = func(t, y, u=u)
    - `y0`   : initial conditions for integration

    Keyword arguments:
    - `method` :  integration scheme in ['euler', 'midpoint', 'rk4'] (default='rk4')
    - `transforms` : a function applied after every step is computed, e.g. wrap_to_pi (default=None)
    """

    tensor_input, func, y0, t = _check_inputs(func, y0, t)
    solver = SOLVERS[method](func, y0, transforms=transforms)
    solution, u_hat, qddot_sol_forward_dyn_tensor, qddot_sol_inv_dyn_tensor, u_tensor = solver.integrate(t, **kwargs)
    if tensor_input:
        solution = solution[0]
    return solution, u_hat, qddot_sol_forward_dyn_tensor, qddot_sol_inv_dyn_tensor, u_tensor


class ActuatedODEWrapper:

    def __init__(self, diffeq):
        """
        Wrapper for compat

        Arguments:
        - `diffeq`: torch.nn.Module that takes q, v, u as arguments
        """
        self.diffeq = diffeq

    def forward(self, delta_t, y, u=None):
        (q, v) = y
        qddot, qddot_sol = self.diffeq(q, v, u, delta_t)
        dy = (v, qddot)
        return dy, qddot_sol

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)

        return getattr(self.diffeq, attr)


SOLVERS = {
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
}


def _assert_increasing(t):
    assert (t[1:] > t[:-1]).all(), 't must be strictly increasing or decrasing'


def _decreasing(t):
    return (t[1:] < t[:-1]).all()


def _check_inputs(func, y0, t):
    if not isinstance(func, ActuatedODEWrapper):
        func = ActuatedODEWrapper(func)

    tensor_input = False
    # print([len(item) for item in y0])
    # print(y0[0].shape)
    if torch.is_tensor(y0):
        tensor_input = True
        y0 = (y0,)

        _base_nontuple_func_ = func
        func = lambda t, y, u: (_base_nontuple_func_(t, y[0], u),)
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'

    for y0_ in y0:
        assert torch.is_tensor(y0_), 'each element must be a torch.Tensor but received {}'.format(
            type(y0_))

    for y0_ in y0:
        if not torch.is_floating_point(y0_):
            raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0_.type()))

    # if not torch.is_floating_point(t):
    #     raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.type()))

    return tensor_input, func, y0, t
