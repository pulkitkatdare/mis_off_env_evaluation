# Installation Instructions
This code uses three modules. It is important to install dependencies related to all of them

1. [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
2. [rl-baselines-zoo ](https://github.com/DLR-RM/rl-baselines3-zoo)
3. [GradientDICE](https://github.com/ShangtongZhang/DeepRL/tree/GradientDICE/deep_rl)

Additionally to run experiments on HalfCheetah, you need to install [Roboschool](https://github.com/openai/roboschool). 
We also used [Sunblaze](https://github.com/sunblaze-ucb/rl-generalization) environments for HalfCheetah to create different Sim2Sim scenarios

# Data Collection Pipiline
Data collection requires a trained policy. To train a policy with respect to any environment. Use the following code 
```
python train.py --algo ppo --env RoboschoolHalfCheetah-v1
```
The experiment creates a ```logs``` folder with the a directory named after the training algorithm. The trained policies are saved in that directory. 

To run data collection pipeline for HalfCheetah. Run the following

```
bash step_one.sh
```
# Beta Training Pipeline

```
bash step_two.sh
```

# OFF ENVIRONMENT EVALUATION ESTIMATION

```
bash step_three.sh
```
