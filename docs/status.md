## Project Summary
In Minecraft, ghasts are large flying mobs that can shoot fireballs at the player. The player can deflect the fireball by hitting it with the correct timing. Here is a video on how it works:

<iframe class="youtube" height="300" src="https://www.youtube.com/embed/sMioimZS_gY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

For our project, our agent will learn how to kill ghasts by deflecting fireballs back at them. Our agent will be given the position and heading of incoming fireballs, and the position of the ghast, and will have to learn where to aim and with what timing to swing to redirect the fireball at the ghast.

## Approach
For training our agent we use the rllib reinforcement learning framework with the Proximal Policy Optimization (PPO) algorithm.

Our current environment consists of an open space with a stationary ghast spawn at a random x position. The agent is made invincible and it is also trapped in a box to prevent knockback since we are mainly training our agent to aim.

### Observation Space
The observation space contains the location and speed of ghasts and fireballs in the environment in addition to the agent’s relative yaw and position.

### Action Space
We converted the agent to a continuous action space since in order to hit the ghast, the agent needs precise actions. Currently there are two actions in the action space: attack and turn. The attack parameter is converted into a binary action since it is not a continuous action. Eventually when we let the ghast freely move we will have to add pitch to our action space.

### Rewards
The agent is rewarded when it is able to redirect a fireball. It is given a higher reward when the fireball is approximately close to the ghast while a huge reward is given when the agent is successful at killing the ghast. The negative rewards are calculated based on how many fireballs the ghast produced in order for the agent to kill the ghast. This rewards the agent to kill the ghast with the least amount of fireballs the ghast produces.

The mission ends when the agent is successful at killing the ghast or if 30 seconds has passed since the beginning of the mission.

## Evaluation
### Quantitative
While there isn’t noticeable improvement in the beginning of training (figure 1), our quantitative evaluation shows that there is slight improvement with our agent after a couple hours of training as shown in this graph (figure 2).

### Qualitative
Though the quantitative results don’t show much improvement, after watching the agent we can see noticeable improvement that the agent accuracy improves in terms of hitting more fireballs, and hitting ghasts more often than its initial training (shown in the demo video). 

## Remaining Goals and Challenges
Currently, for training, we only spawned one stationary ghast per session for the agent to kill. Our plan for the final report is to make the agent work with a moving ghast and possibly multiple ghasts at the same session.

In addition we plan on adjusting the observation space, action space and rewards in order to get better results. We also plan on logging different results in order to better observe the performance of our agent.


## Resources Used
- https://www.youtube.com/watch?v=9RN2Wr8xvro&list=PL-nR3Zo5zPQvaNGqElO9-N-1z-4N94qBi&index=1
- https://microsoft.github.io/malmo/0.14.0/Schemas/MissionHandlers.html
- https://microsoft.github.io/malmo/0.14.0/Documentation/classmalmo_1_1_agent_host.html
- https://docs.ray.io/en/releases-0.8.1/rllib-algorithms.html
- https://arxiv.org/abs/1707.06347
