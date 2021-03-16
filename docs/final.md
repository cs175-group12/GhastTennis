---
layout: default
title: Final Report
---

# Final Report

## Video

<iframe class="youtube" height="300" src="https://www.youtube.com/embed/-GrR4A7EBec" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Project Summary

In Minecraft, ghasts are large flying mobs that can shoot fireballs at the player. The player can deflect the fireball by hitting it with the correct timing. Here is a video on how it works:

<iframe class="youtube" height="300" src="https://www.youtube.com/embed/sMioimZS_gY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

For our project, our agent will learn how to kill ghasts by deflecting fireballs back at them. Our agent will be given the position of the ghasts in the environment and the position & motion of the incoming fireballs. The agent will then learn where to aim and when to swing its punch to redirect the fireball to the ghast.

## Approaches

Initially, we took 2 approaches towards training the AI: a neural network and RLLib’s Proximal Policy Optimization (PPO) algorithm. We later chose the neural network to be our main training approach because it provided better results and shorter training time as opposed to the reinforcement learning approach.

The neural network started out as a Jupyter Notebook version based off of a Youtube tutorial demonstrating digit classification - [Neural Network Explained](https://www.youtube.com/watch?v=9RN2Wr8xvro&list=PL-nR3Zo5zPQvaNGqElO9-N-1z-4N94qBi&index=1). It took a few days to get a version that can predict values, and a few more to get back-propagation working. In the beginning, we tried using different activation functions, but none of them worked well with `NaN` values except for the `sigmoid` function. So, we continued using the `sigmoid` function for the next version. The best accuracy that the early version achieved was 90% with 1 layer.

![](/assets/images/approach1.png)

<p align=”center”>V1’s predict function attempting to use various activation functions with no bias.</p>

For debugging purposes, the neural network was moved into an actual `.py` file so we can use breakpoints. The neural network was redefined to be more general and has a variable shape, making it version 2. Once that was working, one of our team members tried learning PyTorch with the eventual goal of doing evolutionary learning. He quickly discovered that PyTorch was very geared towards gradient descent and copying a network using PyTorch could be huge hassle. In the end, he decided not to use PyTorch and adjusted our already-working network for evolutionary learning, titling it version 3.

![](/assets/images/approach2.png)

<p align=”center>Final version of the predict function in V3.</p>

After the learner was done, we needed a way to accelerate the training process. The `notminecraft.py` script was created with the goal of (from the agent’s perspective) perfectly emulating the observations it would get from real Minecraft. It also allows us to inject scoring logic and run it 46 times faster, per process. A good amount of time was spent to optimize this “virtual world”. Implementing the `quaternionic` Python module improved the speed 100% over `pyquaternion` module, which was used at first to handle rotations. We ran the trained agents in real Minecraft and noticed some bugs that had to be fixed, such as clamping the pitch. The architecture of the simulation is based on Unity (a game engine), with a base `Entity` class that has a `transform`, `start`, and `update` functions. Entities in the simulation are `Fireball`s, `Ghast`s, and the `Agent` itself. The `World` class represents a singular simulation, and due to this, several simulations can be run in parallel using multiprocessing. It also includes utility functions, such as `SphereLineIntersect`, which is used for hit detection against fireballs. 

![](/assets/images/approach3.png)

<p align=”center”>The entities within notminecraft.py.</p>

Initially, there were several ideas on how to do selection and scoring: adding a bonus if it was looking close to the ghast and fireball, punishing it for looking straight up or down, small rewards for almost hitting the fireball, doing boom and bust cycles with the population over time, etc.. In the final version, only hitting a fireball and hitting a ghast with a fireball were rewarded. The selection function was a bit more involved. The array is sorted by score, ascending, and the agents “reproduce” according to $log_2(index) - 1$. For example, in a population of 128 agents, the best ones will reproduce 6 times. The 3rd quartile is 50/50 mutated in place or scrambled, to prevent it from becoming a monoculture.

![](/assets/images/approach4.png)

<p align=”center”>The Natural Selection Algorithm, operating on a list of AI sorted by score.</p>

The biggest hurdles I ran into were overfitting and the lack of randomness no matter what I did from `numpy`. To start with the latter, even when training didn’t utilize sub processes, scores were more indicative of what AI got lucky ghast spawns either right above or right below them - these AI didnt have results that translated between runs and since luck isn’t heritable, they didn’t learn much either. To remedy that I made a list of spawn points, and the ghasts always start at the first then progress through it as they are killed and respawn. Ultimately this training was still not yielding good results though - agents would slowly spin and look up or down , coincidentally getting the first 3 without really responding to input after a literal year of simulated training time. 

I was about ready to give up on neural networks at this point. I even replaced my selection function with complete randomness to see if I could get anything better by pure luck. I didn’t. I decided to sit down and make one by hand to see what an “ideal” network would look like. I discovered that instead of needing hundreds of neurons and 3-5 layers for what I imagined were complex spatial operations, I really only needed 1 layer of the minimum size, sparsely decorated with 1’s and -1s, with a simple bias to determine when to attack.

![](/assets/images/approach5.png)

<p align=”center”>The “Hand Made” Network (scored 275 in the simulation).</p>

In the final 3 trained agents, I set the maximum layer size to the input size, and the maximum number of hidden layers to 2. Instead of having values initialized between $-1/layersize$ and $1/layersize$, they were expanded to be between -1 and 1. The best bot previous to this scored 190 in the simulation. The best bot after that scored 780, and is the one shown in our video.

Still, there was some weirdness that needed sorting out with how Minecraft and Malmo handle space - their coordinate system is strange, -z seems to be forwards. Also Malmo reports pitch and yaw in a way that doesn’t match their documentation. With some tweaking to how input and output are received and fed between the agent and Malmo, we were able to get good performance out of them - though I’m still not 100% sure if it’s indicative of how they performed in the simulation.

When I was about to give up on neural networks I considered a few options that were only partially implemented in version 4 of my neural network: making it an ensemble learner and exchanging members via ‘sexual reproduction’ between the agents that reproduced in the evolutionary learner, or making a computation graph that mutated over time. I didn’t finish these because the simple agent effectively maxed out the score in the simulation, killing at least 13 ghasts in 40 seconds, and killing 12 without missing a fireball.

## Evaluation

Overall I’m more impressed by how well the agents learned in simulation than how well that translated to real Minecraft. Small differences - the small offset from center that the ghast shoots its fireball from, their tendency to slowly run away from the player, imprecise frame timing, input delay while using Malmo, and inconsistent fireball velocity - make a big difference for precise, well performing agents. In addition, Minecraft’s reversed coordinate system was difficult to translate into the agent’s local space, and it’s hard to tell if it’s done correctly. 

I feel as though our agent adequately takes down ghasts, often beating them up, and sometimes hitting their fireballs back into them. The version I made by hand performs a bit better even though it does much worse in the simulation, but I’m glad I was able to internalize the concepts of neural networks well enough to be able to design one on paper that does well.

Ultimately I think our bots limitations stem from the precision and timing required for the task, and the imprecision of Malmo - without a sword you don’t even have 1 server tick to react to an incoming fireball, you need to predict it, and then frame-perfectly attack. This is because Malmo only updates on server tick, instead of queueing inputs per frame like Minecraft does for human players. You can see this in the bot’s wild swinging around as it tries to aim at the ghast - its repeatedly overshooting then overcorrecting until it gets close, when it stabilizes, causing it to miss most of the return shots. The neural network is only a 9x5 matrix, or at most 2 - it’s capable of performing at well above 60fps but because of the limitations of Minecraft it only gets 20. 

### Quantitative

<p align=”center”>Left: Unreliable randomization, rewards from looking towards Ghasts and fireballs.<br>
Right: Deeper neural networks, more random initialization (-1 to 1).</p>

<p align=”center”><img src=”/assets/images/1-l.png”><img src=”/assets/images/1-r.png”></p>

<p align=”center”>Left: Reduced the number of layers from 5-10 to 1-2, reduced maxsize from 783 to 9.<br>
Right: Evaluation time extended from 20s to 40s.</p>

<p align=”center”><img src=”/assets/images/2-l.png”><img src=”/assets/images/2-r.png”></p>

<p align=”center”>Left: Ghast spawns are looped properly instead of throwing errors when all were killed.<br>
Right: Same thing, but `forward` is (0, 0, -1), like real Minecraft.</p>

<p align=”center”><img src=”/assets/images/2-l.png”><img src=”/assets/images/2-r.png”></p>

### Qualitative 

Our simulated Minecraft world showed promising results during training. However, using our trained agent in a real Minecraft environment in Malmo did not translate well. After training high scoring agents in the simulated Minecraft world, we used the trained agent and fed observations from a real Minecraft world and Malmo. Our agent would move in directions not facing the ghast at all. Once we changed a few parameters and retrained another high scoring agent, we noticed that the agent runs up to the ghast and spams attack which does kill the ghast quickly but that was not the results we expected from our agent.

## References

* [Malmo’s Mission Handlers Documentation](https://microsoft.github.io/malmo/0.14.0/Schemas/MissionHandlers.html)

* [Malmo’s Agent Host Documentation](https://microsoft.github.io/malmo/0.14.0/Documentation/classmalmo_1_1_agent_host.html)

* [RLlib Documentation](https://docs.ray.io/en/releases-0.8.1/rllib-algorithms.html)

* [Proximal Policy Optimization Algorithm](https://arxiv.org/abs/1707.06347)

* [Neural Network Explained](https://www.youtube.com/watch?v=9RN2Wr8xvro&list=PL-nR3Zo5zPQvaNGqElO9-N-1z-4N94qBi&index=1)

* [PyTorch in 5 Minutes](https://www.youtube.com/watch?v=nbJ-2G2GXL0&list=PL-nR3Zo5zPQvaNGqElO9-N-1z-4N94qBi&index=1)

* [Which Activation Function Should I Use?](https://www.youtube.com/watch?v=-7scQpJT7uo&list=PL-nR3Zo5zPQvaNGqElO9-N-1z-4N94qBi&index=4)

* [Machine Learning for Video Games](https://www.youtube.com/watch?v=qv6UVOQ0F44&list=PL-nR3Zo5zPQvaNGqElO9-N-1z-4N94qBi&index=3)

* [Paper on Techniques in Deep Learning](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

* [Blogpost on Genetic Algorithms](https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f)
