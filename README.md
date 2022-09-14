# Marginalized Importance Sampling for Off-Environment Policy Evaluation


We investigate how to leverage a good but imperfect model/simulator for off-policy evaluation (OPE) in the framework of marginalized importance sampling (MIS). Existing MIS methods learn a density ratio between the occupancies of the target and the behavior policies, and face two difficulties *simultaneously*:
1. *large ratios*: the density ratio can largely deviate from 1, and 
2. *indirect supervision*:  no sample from the target policies' occupancy is *directly* available, and the ratio needs to be inferred *indirectly* via the Bellman flow equation. 

In this paper, we propose a new MIS method that address the above challenges. By introducing the target policy's occupancy in the simulator as an intermediate variable, our method splits the ratio into the product of two terms and learns them separately, where the first term enjoys *direct supervision* and the second term has *small magnitude* (i.e., close to 1), and none of the terms are subject to both difficulties simultaneously. A sample complexity analysis is provided and offers insights about error propagation in the two-step estimation procedure, and we provide additional results and empirical evaluation of our method on Sim2Sim examples like Cartpole and Roboschool environments. We also demonstrate practical applications of our model on Sim2Real validation on KinovaGen3

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
