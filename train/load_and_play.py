#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import random
import time

import torch
#from vmas.simulator.utils import EvaluationUtils, PathUtils

from vmas import make_env
from ray.rllib.agents.ppo import PPOTrainer
from vmas.simulator.utils import save_video

def use_vmas_env(render: bool = True, save_render: bool = True):
    assert not (save_render and not render), "To save the video you have to render it"

    
    #scenario_name = "give_way_test"
    #scenario_name = "ball_trajectory"
    scenario_name = "multi_give_way_1"

    
    n_agents = 4

    num_envs = 32  # Number of vectorized environments
    continuous_actions = True
    device = "cpu"  # or cuda or any other torch device
    n_steps = 300  # Number of steps before returning done
    num_vectorized_envs = 96
    dict_spaces = True  # Weather to return obs, rewards, and infos as dictionaries with agent names
    # (by default they are lists of len # of agents)
	
    """simple_2d_action_1 = (
        [0., -0.5] if continuous_actions else [1]
    )  # Simple action tell each agent to go down
    simple_2d_action_2 = (
        [-0.5, 0.] if continuous_actions else [1]
    )  # Simple action tell each agent to go down
    simple_2d_action_3 = (
        [0.0, 0.5] if continuous_actions else [1]
    )  # Simple action tell each agent to go down
    simple_2d_action_4 = (
        [0.5, 0.] if continuous_actions else [1]
    )  # Simple action tell each agent to go down
    """
    
    
    simple_2d_action_1 = (
        [0.5, 0.00] if continuous_actions else [1]
    )  # Simple action tell each agent to go down
    simple_2d_action_2 = (
        [-0.5, 0.] if continuous_actions else [1]
    )  # Simple action tell each agent to go down
    simple_2d_action_3 = (
        [0.5, 0.00] if continuous_actions else [1]
    )  # Simple action tell each agent to go down
    simple_2d_action_4 = (
        [-0.5, 0.] if continuous_actions else [1]
    )  # Simple action tell each agent to go down

    env = make_env(
        scenario=scenario_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        dict_spaces=dict_spaces,
        wrapper=None,
        seed=None,
        # Environment specific variables
        n_agents=n_agents,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    for s in range(n_steps):
        step += 1
        print(f"Step {step}")
        # print(f'save videos:{save_render}!')

        # VMAS actions can be either a list of tensors (one per agent)
        # or a dict of tensors (one entry per agent with its name as key)
        # Both action inputs can be used independently of what type of space its chosen
        dict_actions = random.choice([True, False])

        actions = {} if dict_actions else []
        for i, agent in enumerate(env.agents):
            print(f"agent_No:{i}")
            if i == 1:
                action = torch.tensor(
                simple_2d_action_1,
                device=device,
                ).repeat(num_envs, 1)
                if dict_actions:
                    actions.update({agent.name: action})
                else:
                    actions.append(action)
            elif i == 2:
                action = torch.tensor(
                simple_2d_action_2,
                device=device,
                ).repeat(num_envs, 1)
                if dict_actions:
                    actions.update({agent.name: action})
                else:
                    actions.append(action)
            elif i == 3:
                action = torch.tensor(
                simple_2d_action_3,
                device=device,
                ).repeat(num_envs, 1)
                if dict_actions:
                    actions.update({agent.name: action})
                else:
                    actions.append(action)
            else:
                action = torch.tensor(
                simple_2d_action_4,
                device=device,
                ).repeat(num_envs, 1)
                if dict_actions:
                    actions.update({agent.name: action})
                else:
                    actions.append(action)  
            
            """action = torch.tensor(
                simple_2d_action_1,
                device=device,
            ).repeat(num_envs, 1)
            if dict_actions:
                actions.update({agent.name: action})
            else:
                actions.append(action)
            """

        obs, rews, dones, info = env.step(actions)

        if render:
            frame = env.render(
                mode="rgb_array" if save_render else "human",
                agent_index_focus=None,  # Can give the camera an agent index to focus on
                visualize_when_rgb=True,
            )
            if save_render:
                frame_list.append(frame)

    if render and save_render:
        
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )


if __name__ == "__main__":
    use_vmas_env(render=True, save_render=True)
