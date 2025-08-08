import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_folder)

import torch
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from dynamics.freeflyer import FreeflyerModel, ocp_no_obstacle_avoidance, ocp_obstacle_avoidance, check_koz_constraint
from optimization.ff_scenario import n_obs, n_time_rpod, obs, table, robot_radius, safety_margin, dt, T
# import decision_transformer.managestoch as DT_manager # use this for normal-huggingface attention inference
import decision_transformer.manageflashstoch as DT_manager # use this for flash attention inference
import time

'''
To use flash attention, uncomment the following line:
    import decision_transformer.manageflashstoch as DT_manager

To use the previous version of the model, uncomment the following line:
    import decision_transformer.managestoch as DT_manager
'''

# Simulation configuration
transformer_model_name =  'checkpoint_ff_flash_art_stoch_changed'# 'checkpoint_ff_flash_art_stoch_dynamic_std' # flash model 'checkpoint_ff_noflash_art_stoch_dynamic_std'
# transformer_model_name =  'checkpoint_ff_art_stoch' #'checkpoint_ff_art_stoch' # regular model 
transformer_config = DT_manager.transformer_import_config(transformer_model_name)
mdp_constr = transformer_config['mdp_constr']
timestep_norm = transformer_config['timestep_norm']
transformer_ws = 'dyn'
datasets, dataloaders = DT_manager.get_train_val_test_data(mdp_constr=mdp_constr, timestep_norm=timestep_norm)
train_dataset, val_dataset, test_dataset = datasets
train_loader, eval_loader, test_loader = dataloaders

'''
Possible options for initial coditions:
Init: np.array([0.22, 2.14, 1.53, 0, 0, 0])
      np.array([1.58643470e-01,  5.06685234e-01, -1.07279630e+00, 0.0,0.0,0.0])
      np.array([0.29490604, 2.34616295, 1.69898474,0,0,0])
Final: np.array([3.31, 0.60, -1.28, 0, 0, 0])
       np.array([3.25586406,  2.29016048, -0.80894559, 0.0, 0.0, 0.0])
       np.array([3.15600952, 0.25998125, 0.99323413, 0,0,0])
'''
state_init = np.array([0.22, 2.14, 1.53, 0, 0, 0]) 
state_final = np.array([3.31, 0.60, -1.28, 0, 0, 0])
# test_sample = next(iter(test_loader))
data_stats = test_loader.dataset.data_stats
# test_sample[0][0,:,:] = (torch.tensor(np.repeat(state_init[None,:], K, axis=0)) - data_stats['states_mean'])/(data_stats['states_std'] + 1e-6)#(torch.tensor(xs[:-1,:]) - data_stats['states_mean'])/(data_stats['states_std'] + 1e-6)#
# test_sample[1][0,:,:] = torch.zeros((K,3))#(torch.tensor(us) - data_stats['actions_mean'])/(data_stats['actions_std'] + 1e-6)#
# test_sample[2][0,:,0] = torch.zeros((K,))#torch.from_numpy(compute_reward_to_go(test_sample[1][0,:,:]))#
# test_sample[3][0,:,0] = torch.zeros((K,))#torch.from_numpy(compute_constraint_to_go(test_sample[0][0,:,:].cpu().numpy(), obs_positions, obs_radii))#
# test_sample[4][0,:,:] = (torch.tensor(np.repeat(state_final[None,:], K, axis=0)) - data_stats['goal_mean'])/(data_stats['goal_std'] + 1e-6)


select_idx = True # set to True to manually select a test trajectory via its index (idx)
idx = 1185 #22260 # index of the test trajectory 19030, 11787, 5514->infeasible in cvxMPC at 4 timestep 32172
# Sample from test dataset
if select_idx:
    test_sample = test_loader.dataset.getix(idx)
else:
    test_sample = next(iter(test_loader))
if mdp_constr:
    states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = test_sample
else:
    states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = test_sample


data_stats = test_loader.dataset.data_stats
state_init = ((test_sample[0][0,0,:] * data_stats['states_std'][0]) + (data_stats['states_mean'][0])).cpu().numpy()
state_final = ((test_sample[4][0,0,:] * data_stats['goal_std'][0]) + (data_stats['goal_mean'][0])).cpu().numpy()

# FreeFlyerModel
ffm = FreeflyerModel(verbose=True)
dt = dt.item()
time_sec = np.hstack((time_sec[0,0], time_sec[0,0,-1] + dt))


# Warmstarting and optimization
# Obstacles info
obs_positions = obs['position']
obs_radii = (obs['radius'] + robot_radius)*safety_margin
# Solve Convex Problem
traj_cvx, _, n_iter_cvx, feas_cvx = ocp_no_obstacle_avoidance(ffm, state_init, state_final)
states_ws_cvx, actions_ws_cvx = traj_cvx['states'], traj_cvx['actions_G']
print('CVX cost:', np.sum(la.norm(actions_ws_cvx, ord=1, axis=0)))
constr_cvx, constr_viol_cvx= check_koz_constraint(states_ws_cvx.T, obs_positions, obs_radii)
# Solve SCP
traj_scp_cvx, J_vect_scp_cvx, iter_scp_cvx, feas_scp_cvx = ocp_obstacle_avoidance(ffm, states_ws_cvx+np.array([0,0.,0,0,0,0]).reshape(-1,1), actions_ws_cvx, state_init, state_final)
states_scp_cvx, actions_scp_cvx = traj_scp_cvx['states'], traj_scp_cvx['actions_G']
print('SCP cost:', np.sum(la.norm(actions_scp_cvx, ord=1, axis=0)))
print('J vect', J_vect_scp_cvx)
constr_scp_cvx, constr_viol_scp_cvx = check_koz_constraint(states_scp_cvx.T, obs_positions, obs_radii)

# Import the Transformer
model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader)
print(model)
model.eval()
inference_func = getattr(DT_manager, 'torch_model_inference_'+transformer_ws)
print('Using ART model \'', transformer_model_name, '\' with inference function DT_manage.'+inference_func.__name__+'()')
rtg = - np.sum(la.norm(actions_ws_cvx, ord=1, axis=0)) if mdp_constr else None #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DT_trajectory, runtime_DT = inference_func(model, test_loader, test_sample, rtg_perc=1., ctg_perc=0., rtg=rtg, ctg_clipped=True)
print('ART runtime:', runtime_DT)


###############################################################
### for batch trajectory plotting
############################################################### 

n_trajectories = DT_trajectory['xypsi_' + transformer_ws].shape[0]
states_ws_DT_list = []
actions_ws_DT_list = []
feasible_DT_list = np.full(n_trajectories, True, dtype=bool)
fig, ax = plt.subplots(figsize=(12, 8))
k = 0
for i in range(n_trajectories):
    # Extract single trajectory
    xypsi_i = DT_trajectory['xypsi_' + transformer_ws][i]     # shape (100, 6)
    dv_i = DT_trajectory['dv_' + transformer_ws][i]            # shape (100, 3)

    # Compute warm-start state (append final state + B_imp @ final action)
    final_state = (xypsi_i[-1] + ffm.B_imp @ dv_i[-1]).reshape((1, 6))
    states_ws = np.append(xypsi_i, final_state, axis=0).T      # shape (6, 101)
    actions_ws = dv_i.T                                        # shape (3, 100)

    states_ws_DT_list.append(states_ws)
    actions_ws_DT_list.append(actions_ws)

    # Compute cost and constraint
    print(f'Traj {i} ART cost:', np.sum(la.norm(actions_ws, ord=1, axis=0)))
    constr, constr_viol = check_koz_constraint(states_ws.T, obs_positions, obs_radii)

    # Optional: solve SCP for each trajectory (commented out for speed)
    traj_scp, J_vect_scp, iter_scp, feas_scp = ocp_obstacle_avoidance(ffm, states_ws, actions_ws, state_init, state_final)
    
    states_scp = traj_scp['states']
    actions_scp = traj_scp['actions_G']
    if actions_scp is not None:
        k += 1
        print(f'Traj {i} SCP cost:', np.sum(la.norm(actions_scp, ord=1, axis=0)))
        # print(f'Traj {i} J vect', J_vect_scp)
        # constr_scp, constr_viol_scp = check_koz_constraint(states_scp.T, obs_positions, obs_radii)
        print(f'Traj {i} SCP feasible:', {k})
        ax.plot(states_scp[0,:], states_scp[1,:], 'c', alpha=0.3, linewidth=1)
    else:
        feasible_DT_list[i] = False
        print(f'Traj {i} SCP infeasible')
    # Plot warm-start trajectory
    ax.plot(states_ws[0, :], states_ws[1, :], color=[0.5, 0.5, 0.5], alpha=0.5, linewidth=1.0, zorder=3)

# Table and obstacles
ax.add_patch(Rectangle((0, 0), table['xy_up'][0], table['xy_up'][1], fc=(0.5, 0.5, 0.5, 0.2), ec='k', label='table', zorder=2.5))
for n_obs in range(obs['radius'].shape[0]):
    label_obs = 'obs' if n_obs == 0 else None
    label_robot = 'robot radius' if n_obs == 0 else None
    ax.add_patch(Circle(obs['position'][n_obs, :], obs['radius'][n_obs], fc='r', label=label_obs, zorder=2.5))
    ax.add_patch(Circle(obs['position'][n_obs, :], obs['radius'][n_obs] + robot_radius, fc='r', alpha=0.2, label=label_robot, zorder=2.5))

# Initial and final state markers
ax.scatter(state_init[0], state_init[1], label='state init', zorder=3)
ax.scatter(state_final[0], state_final[1], label='state final', zorder=3)

# Axis, labels, etc.
ax.set_aspect('equal')
ax.set_xlabel('X [m]', fontsize=10)
ax.set_ylabel('Y [m]', fontsize=10)
ax.grid(True)
ax.legend(loc='best', fontsize=10)

# Save plot
plt.savefig(root_folder + '/optimization/saved_files/plots/pos_3d_multi.png')


# plot the actions from the CVX and ART ws solution along each action dimension sperately

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)  # 3 subplots in one column
for i in range(n_trajectories):
    actions_ws_DT = actions_ws_DT_list[i]
    for j in range(actions_ws_DT.shape[0]):
        axes[j].plot(actions_ws_DT[j, :], alpha=0.4, label=f'Traj {i+1}' if j == 0 else "")  # Label only once to avoid clutter
# Plot the CVX warm-start actions on each subplot
line_styles = ['-', '-', '-']
for j in range(3):
    axes[j].plot(actions_ws_cvx[j, :], 'k' + line_styles[j], linewidth=2.0, label='CVX Warm-start')

# Add titles, labels, grid, and legend
for j in range(3):
    axes[j].set_title(f'Action Dimension {j+1}', fontsize=12)
    axes[j].set_xlabel('Time Steps', fontsize=10)
    axes[j].grid(True)
    if j == 0:
        axes[j].set_ylabel('Action Value', fontsize=10)
    axes[j].legend(fontsize=9)

plt.tight_layout()
plt.savefig(root_folder + '/optimization/saved_files/plots/actions_multi.png')

###############################################################
##### single trajectory plotting
############################################################### 

# # states_ws_DT = np.append(DT_trajectory['xypsi_' + transformer_ws], (DT_trajectory['xypsi_' + transformer_ws][:,-1] + ffm.B_imp @ DT_trajectory['dv_' + transformer_ws][:, -1]).reshape((6,1)), 1)# set warm start
# # actions_ws_DT = DT_trajectory['dv_' + transformer_ws]
# # #############Pick the first trajectory (index 0)
# xypsi_single = DT_trajectory['xypsi_' + transformer_ws][0]   # shape: (100, 6)
# dv_single = DT_trajectory['dv_' + transformer_ws][0]         # shape: (100, 3)

# # Compute warm-start trajectory (append final state + B_imp @ final action)
# states_ws_DT = np.append(xypsi_single.T, (xypsi_single[-1] + ffm.B_imp @ dv_single[-1]).reshape((6, 1)), axis=1)
# actions_ws_DT = dv_single.T

# print('ART cost:', np.sum(la.norm(actions_ws_DT, ord=1, axis=0)))
# print('ART runtime:', runtime_DT)
# constr_DT, constr_viol_DT = check_koz_constraint(states_ws_DT.T, obs_positions, obs_radii)

# # Solve SCP
# traj_scp_DT, J_vect_scp_DT, iter_scp_DT, feas_scp_DT = ocp_obstacle_avoidance(ffm, states_ws_DT, actions_ws_DT, state_init, state_final)
# states_scp_DT, actions_scp_DT = traj_scp_DT['states'], traj_scp_DT['actions_G']
# print('SCP cost:', np.sum(la.norm(actions_scp_DT, ord=1, axis=0)))
# print('J vect', J_vect_scp_DT)
# constr_scp_DT, constr_viol_scp_DT = check_koz_constraint(states_scp_DT.T, obs_positions, obs_radii)

# # Plotting

# # 3D position trajectory
# ax = plt.figure(figsize=(12,8)).add_subplot()
# p1 = ax.plot(states_ws_cvx[0,:], states_ws_cvx[1,:], 'k', linewidth=1.5, label='warm-start cvx', zorder=3)
# p2 = ax.plot(states_scp_cvx[0,:], states_scp_cvx[1,:], 'b', linewidth=1.5, label='scp-cvx', zorder=3)
# p3 = ax.plot(states_ws_DT[0,:], states_ws_DT[1,:], c=[0.5,0.5,0.5], linewidth=1.5, label='warm-start ART-' + transformer_ws, zorder=3)
# # p4 = ax.plot(states_scp_DT[0,:], states_scp_DT[1,:], 'c', linewidth=1.5, label='scp-ART-' + transformer_ws, zorder=3)
# ax.add_patch(Rectangle((0,0), table['xy_up'][0], table['xy_up'][1], fc=(0.5,0.5,0.5,0.2), ec='k', label='table', zorder=2.5))
# for n_obs in range(obs['radius'].shape[0]):
#     label_obs = 'obs' if n_obs == 0 else None
#     label_robot = 'robot radius' if n_obs == 0 else None
#     ax.add_patch(Circle(obs['position'][n_obs,:], obs['radius'][n_obs], fc='r', label=label_obs, zorder=2.5))
#     ax.add_patch(Circle(obs['position'][n_obs,:], obs['radius'][n_obs]+robot_radius, fc='r', alpha=0.2, label=label_robot, zorder=2.5))
# ax.scatter(state_init[0], state_init[1], label='state init', zorder=3)
# ax.scatter(state_final[0], state_final[1], label='state final', zorder=3)
# ax.grid(True)
# ax.set_aspect('equal')
# ax.set_xlabel('X [m]', fontsize=10)
# ax.set_ylabel('Y [m]', fontsize=10)
# ax.grid(True)
# ax.legend(loc='best', fontsize=10)
# plt.savefig(root_folder + '/optimization/saved_files/plots/pos_3d.png')

# # Constraint satisfaction
# plt.figure()
# plt.plot(time_sec, constr_cvx.T, 'k', linewidth=1.5, label='warm-start cvx')
# plt.plot(time_sec, constr_scp_cvx.T, 'b', linewidth=1.5, label='scp-cvx')
# plt.plot(time_sec, constr_DT.T, c=[0.5,0.5,0.5], linewidth=1.5, label='warm-start ART-' + transformer_ws)
# # plt.plot(time_sec, constr_scp_DT.T, 'c', linewidth=1.5, label='scp-ART-' + transformer_ws)
# plt.plot(time_sec, np.zeros(n_time_rpod+1), 'r-', linewidth=1.5, label='koz')
# plt.xlabel('time [orbits]', fontsize=10)
# plt.ylabel('keep-out-zone constraint [-]', fontsize=10)
# plt.grid(True)
# plt.legend(loc='best', fontsize=10)
# plt.savefig(root_folder + '/optimization/saved_files/plots/constr.png')
