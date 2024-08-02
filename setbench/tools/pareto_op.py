import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.utils.multi_objective import pareto
import wandb

def pareto_frontier(solutions, rewards, maximize=False):
    if len(rewards.shape) == 2 and rewards.shape[0] == 1:
        return solutions, rewards
    pareto_mask = pareto.is_non_dominated(torch.tensor(rewards) if maximize else -torch.tensor(rewards))
    if len(rewards.shape) == 2:
        pareto_front = solutions[pareto_mask] if solutions is not None else None
        pareto_rewards = rewards[pareto_mask]
    elif len(rewards.shape) == 3:
        pareto_front = [solutions[pareto_mask[i]] for i in range(pareto_mask.shape[0])] if solutions is not None else None
        pareto_rewards = [rewards[i][pareto_mask[i]] for i in range(pareto_mask.shape[0])]
    return pareto_front, pareto_rewards

def plot_pareto(pareto_rewards, all_rewards, pareto_only=False):
    if pareto_rewards.shape[-1] < 3:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not pareto_only:
            ax.scatter(*np.hsplit(all_rewards, all_rewards.shape[-1]), color="grey", label="All Samples")
        ax.scatter(*np.hsplit(pareto_rewards, pareto_rewards.shape[-1]), color="red", label="Pareto Front")
        ax.set_xlabel("Reward 1")
        ax.set_ylabel("Reward 2")
        ax.legend()
        return wandb.Image(fig)
    if pareto_rewards.shape[-1] == 3:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter3d(
            x=all_rewards[:, 0],
            y=all_rewards[:, 1],
            z=all_rewards[:, 2],
            mode='markers',
            marker_color="grey",
            name="All Samples"
        ),
        go.Scatter3d(
            x=pareto_rewards[:, 0],
            y=pareto_rewards[:, 1],
            z=pareto_rewards[:, 2],
            mode='markers',
            marker_color="red",
            name="Pareto Front"
        )])
        fig.update_traces(marker=dict(size=8),
                  selector=dict(mode='markers'))
        return fig

def thermometer(v, n_bins=50, vmin=0, vmax=32):
    device = v.device
    bins = torch.linspace(vmin, vmax, n_bins).to(device)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap

def test_pareto():
    # wandb.init(project="pareto_test")
    solutions = torch.LongTensor([
        [1, 2, 3],
        [2, 1, 4],
        [3, 3, 2],
        [4, 4, 1],
        [5, 5, 5],
    ])
    rewards = torch.LongTensor([
        [1, 5],
        [2, 2],
        [3, 3],
        [4, 3],
        [5, 1],
    ])
    rewards2 = torch.LongTensor([
        [1, 5, 4],
        [2, 2, 2],
        [3, 3, 2],
        [4, 3, 4],
        [5, 1, 4],
    ])
    rewards3 = torch.LongTensor([
        [1, 5, 4],
        [0, 2, 2],
        [0, 3, 2],
        [0, 3, 4],
        [5, 1, 4],
    ])
    rewards4 = torch.cat([rewards2.unsqueeze(0), rewards3.unsqueeze(0)], dim=0)

    pareto_front, pareto_rewards = pareto_frontier(solutions, rewards)
    print(solutions, rewards)
    print(pareto_front, pareto_rewards)

    # fig = plot_pareto(pareto_rewards, rewards)
    # wandb.log({'pareto_front':fig})

    pareto_front, pareto_rewards = pareto_frontier(solutions, rewards2)
    print(solutions, rewards)
    print(pareto_front, pareto_rewards)

    # fig2 = plot_pareto(pareto_rewards, rewards2)
    # wandb.log({'pareto_front2':fig2})

    pareto_front, pareto_rewards = pareto_frontier(solutions, rewards4)
    print(solutions, rewards)
    print(pareto_front, pareto_rewards)

def test_thermometer():
    v = torch.tensor([0.2,0.56,0.81])
    print(thermometer(v, 50, 0, 1))

if __name__ == "__main__":
    test_pareto()
    # test_thermometer()