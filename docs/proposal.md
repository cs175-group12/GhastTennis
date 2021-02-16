---
layout: default
title:  Proposal
---

## Summary of the Project

In Minecraft, ghasts are large flying mob that can shoot fireballs at the player. The player can deflect the fireball by hitting it with the correct timing. Here is a video on how it works:

<iframe class="youtube" height="300" src="https://www.youtube.com/embed/sMioimZS_gY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

For our project, our agent will learn how to kill ghasts by deflecting fireballs back at them. Our agent will be given the position and heading of incoming fireballs, and the position of the ghast, and will have to learn where to aim and with what timing to swing to redirect the fireball at the ghast. 

## AI/ML Algorithms

We will use Q-learning algorithm to maximize the agent score.

## Evaluation Plan

We will train our agent in a flat environment with at least one ghast. The agent will be invulnerable during training to prevent it from dying before the mission ends. We will reward our agent based off whether the agent is able to hit the ghast or, if it misses, we will reward the agent based on the distance of how close it was to hitting the ghast. It will also be rewarded for aiming at the incoming fireball, swinging closer to the correct timing, and hitting the fireball. We hope that with these metrics the agent will learn to close the distance of how much it misses in order to eventually learn to hit ghasts consistently.

Ideally we want our agent be consistent in deflecting the fireballs and be able to kill multiple ghasts in one session. Our moonshot case would be to put our agent in a real minecraft environment and test how well it is at killing ghasts there.

## Appointment with the Instructor

Friday January 22, 2021 3:30pm 

## Weekly Meeting Times

Monday/Wednesday 7pm

## Plan for status report

At minimum, we hope to have our agent be able to deflect a stationary ghast's fireball in a flat pre made environment. 
