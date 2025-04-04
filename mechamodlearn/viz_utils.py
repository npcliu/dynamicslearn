#
# File: viz_utils.py
#
import matplotlib.pyplot as plt
import torch

from mechamodlearn.odesolver import odeint
from mechamodlearn import utils


def plot_traj(system_x_T_B, model_x_T_B, tstar):
    D = system_x_T_B.shape[-1]
    cm = plt.get_cmap('tab10')
    fig, axs = plt.subplots(1, D, figsize=(6 * D, 6), squeeze=False)
    for d in range(D):  # For each DoF
        for i in range(system_x_T_B.shape[1]):
            axs[0, d].plot(tstar[:, i], system_x_T_B[:, i, d], color=cm(i), label='True {}'.format(i),
                           alpha=0.8)
            axs[0, d].plot(tstar[:, i], model_x_T_B[:, i, d], color=cm(i), ls='--',
                           label='Pred {}'.format(i), alpha=0.8)

        axs[0, d].set_xlabel('$t$')
        axs[0, d].set_ylabel('$x_{}$'.format(d))
        axs[0, d].legend(frameon=False)

    return fig


def vizqvmodel(model, q_B_T, v_B_T, ddq_B_T, u_B_T, t_B_T, method='rk4'):
    B = q_B_T.size(0)
    # TODO:画图时这些张量或元组的维度还有问题，要改
    q_T_B = q_B_T.transpose(1, 0)
    v_T_B = v_B_T.transpose(1, 0)
    ddq_T_B = ddq_B_T.transpose(1, 0)
    u_T_B = u_B_T.transpose(1, 0)
    t_T_B = t_B_T.transpose(1, 0)
    with torch.no_grad():
        # Simulate forward
        model.reset_buffer(q_T_B[0], u_T_B[0])
        solution, u_hat_T_B, qddot_tensor, _, _ = odeint(model, (q_T_B[0], v_T_B[0]), (q_T_B, v_T_B, ddq_T_B),
                                                         t_T_B, u=u_T_B, method=method,
                                        transforms=(lambda x: utils.wrap_to_pi(x, model.thetamask),
                                                    lambda x: x))
        qpreds_T_B_, vpreds_T_B_ = solution
        qpreds_T_B = qpreds_T_B_[:-1]
        vpreds_T_B = vpreds_T_B_[:-1]
        qpreds_T_B = utils.wrap_to_pi(qpreds_T_B.view(-1, model._qdim), model.thetamask).view(
            -1, B, model._qdim)

    q_fig = {
        'qtraj':
            plot_traj(q_T_B[:, 0:2, 0:5].detach().cpu().numpy(),
                      qpreds_T_B[:, 0:2, 0:5].detach().cpu().numpy(), t_T_B[:, 0:2, 0].detach().cpu().numpy())
    }
    v_fig = {
        'vtraj':
            plot_traj(v_T_B[:, 0:2, 0:5].detach().cpu().numpy(),
                      vpreds_T_B[:, 0:2, 0:5].detach().cpu().numpy(), t_T_B[:, 0:2, 0].detach().cpu().numpy())
    }
    u_fig = {
        'utraj':
            plot_traj(u_T_B[0:99, 0:2, 0:5].detach().cpu().numpy(),
                      u_hat_T_B[:, 0:2, 0:5].detach().cpu().numpy(), t_T_B[0:99, 0:2, 0].detach().cpu().numpy())
    }
    qddot_fig = {
        'utraj':
            plot_traj(ddq_T_B[0:99, 0:2, 0:5].detach().cpu().numpy(),
                      qddot_tensor[0:99, 0:2, 0:5].detach().cpu().numpy(), t_T_B[0:99, 0:2, 0].detach().cpu().numpy())
    }

    return {**q_fig, **v_fig}
