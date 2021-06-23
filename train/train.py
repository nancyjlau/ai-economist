import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ai_economist import foundation
from proj.config import env_config
from proj.ppo import PPO, Memory
from tutorials.utils import plotting

from IPython import display

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_interval = 10
max_episodes = 50000
graphical_episode_log_frequency = 1
n_latent_var = 512
update_timestep = 2000
lr = 0.01
betas = (0.9, 0.999)
gamma = 0.99
K_epochs = 4
eps_clip = 0.2
random_seed = None
plot_every = 2


def sample_random_action(agent, mask):  # For planner
    """Sample random UNMASKED action(s) for agent."""
    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
        # print(len(split_masks))
        # print(split_masks[0] / split_masks[0].sum())
        return [np.random.choice(np.arange(len(m_)), p=m_ / m_.sum()) for m_ in split_masks]

    # Return a single action
    else:
        return np.random.choice(np.arange(agent.action_spaces), p=mask / mask.sum())


def sample_random_actions(env, obs):
    """Samples random UNMASKED actions for each agent in obs."""

    actions = {
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    return actions


def main():
    log_dir = Path(__file__).parent / f"exp{int(time.time())}"
    log_dir.mkdir()
    env = foundation.make_env_instance(**env_config)
    state = env.reset()

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = [Memory() for _ in range(env.n_agents)]

    action_dim = state['0']["action_mask"].size  # todo mask tells which action cannot be taken
    state_dim = state['0']["flat"].size

    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, device)

    # logging variables
    running_reward = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        if i_episode % graphical_episode_log_frequency == 0:
            obs = env.reset(force_dense_logging=True)
        else:
            obs = env.reset()

        for t in range(env.episode_length):
            time_step += 1
            actions = sample_random_actions(env, obs)  # Initialize dict with random actions then fill with selected values

            for agent_id in range(env.n_agents):
                agent_id_str = str(agent_id)
                memory_agent = memory[agent_id]
                action_mask = torch.tensor(state[agent_id_str]["action_mask"], device=device)
                agent_state = state[agent_id_str]["flat"]
                action = ppo.policy_old.act(agent_state, memory_agent, action_mask)
                actions[agent_id_str] = action

            state, reward, done, info = env.step(actions)
            # Saving reward and is_terminals:
            for agent_id in range(env.n_agents):
                agent_id_str = str(agent_id)
                memory_agent = memory[agent_id]
                agent_reward = -reward[agent_id_str]
                memory_agent.rewards.append(agent_reward)
                memory_agent.is_terminals.append(done)
                running_reward += agent_reward

                # update if its time
                if time_step % update_timestep == 0:
                    ppo.update(memory_agent)
                    memory_agent.clear_memory()
                    time_step = 0
            if done['__all__']:
                break


        # save every 500 episodes
        if i_episode % 100 == 0:
            torch.save(ppo.policy.state_dict(), log_dir / f'ckpt-{i_episode}.pth')

        # logging
        if i_episode % log_interval == 0:
            running_reward = float((running_reward / log_interval))

            print(f'Episode {i_episode} \t Avg reward: {running_reward}')
            running_reward = 0

        if i_episode % graphical_episode_log_frequency == 0:
            dense_log = env.previous_episode_dense_log
            (fig0, fig1, fig2), incomes, endows, c_trades, all_builds = plotting.breakdown(dense_log)
            print(f"Incomes = {incomes}, Endows = {endows}, c_trades = {c_trades}, all_builds = {all_builds}")
            # fig0.savefig(log_dir / f"fig0-{i_episode}.png", dpi=fig0.dpi)
            fig1.savefig(log_dir / f"fig1-{i_episode:04d}.png", dpi=fig1.dpi)
            fig2.savefig(log_dir / f"fig2-{i_episode:04d}.png", dpi=fig2.dpi)
            plt.close(fig0)
            plt.close(fig1)
            plt.close(fig2)
            with open(log_dir / f'logs-{i_episode:04d}.pickle', 'wb') as handle:
                pickle.dump(dense_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        if ((i_episode + 1) % plot_every) == 0:
            do_plot(env, ax, fig)

def do_plot(env, ax, fig):
    plotting.plot_env_state(env, ax)
    ax.set_aspect('equal')
    display.display(fig)
    return

if __name__ == '__main__':
    main()
