## Summary of the Project

![](https://static.wikia.nocookie.net/minecraft_gamepedia/images/d/d5/Ghast_JE2_BE2.gif)

In Minecraft, ghasts are large flying mob that can shoot fireballs at the player. The player can deflect the fireball by hitting it with the correct timing. 

For our project, our agent will learn how to kill ghasts by deflecting fireballs back at them. Our agent will have to learn how to recognize the incoming fireballs and be aware of the location of ghasts in order to accurately aim and shoot down ghasts.

## AI/ML Algorithms

We will use an image recognition algorithm to detect the ghast fireballs and the Q-learning algorithm to maximize the agent score.

## Evaluation Plan

We will reward our agent based off whether the agent is able to hit the ghast or if it misses we will reward the agent based on the distance of how close it was to hitting the ghast. We hope that with this metric the agent will learn to close the distance of how much it misses in order to eventually learn to hit ghasts consistently. In addition, if the agent dies, we will subtract points to deter the agent from making bad decisions.

Ideally we want our agent be consistent in killing ghast and be able to kill multiple ghasts in one session without dying. Our moonshot case would be to put our agent in a real minecraft environment and test how well it is at killing ghasts there.

## Appointment with the Instructor

Friday January 22, 2021 3:30pm 