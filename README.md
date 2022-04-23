# MARL_IS_Project
Intelligent Systems Class Project, Team 1

## Discription
Using a vacuum agent to elimate dirt object. Using Q-Learning as the main form of Reinforment learning to solve this problem.

## MovingVacuumStaticMold.py
First iteration of the project to test our initial environment with a moving vacuum agent tasked to find a stationary mold object.

## MovingVacuumMovingMold.py
Second iteration of the project to test our environment with a moving vacuum agent tasked to find a moving mold object.

## QLearningVacuumQLearningMold.py
Third and final iteration of the project to test our environment with a Q-Learning Vacuum agent facing a Q-Learning Mold agent using a Mini-max algorithm to minipulate their actions from the Q-table.

## Setup
Either create a new environment for python by using Anaconda or by any other means. By using the ***pip*** command in either your environment console or in your main console where python is set to, download the required dependencies below. The numpy library is also required for the program to work and it comes packaged with ***opencv-python***. After you have an enviornment with the dependencies installed you can then run each python script in your system. Notice the comments after each global variable. The values can be changed to further test the agents environment after the agent has been trained.

## Dependencies
opencv-python
```
pip install opencv-python
```
pillow
```
pip install Pillow
```
matplotlib
```
pip install matplotlib
```
