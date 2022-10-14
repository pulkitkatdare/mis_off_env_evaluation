mkdir offline_data
var_param="env_id:1"
timesteps=150
echo $var_param
for policy_noise in 0.4 0.6
do 
python enjoy.py --env-kwargs "$var_param" --algo ppo --env RoboschoolHalfCheetah-v1 -f logs/ --exp-id 1 --load-best --save_file 'offline_data/roboschoolhalfcheetah' --timesteps $timesteps -n 10000 \
--policy_noise $policy_noise
done 
done 

var_param="env_id:0"
timesteps=150
echo $var_param
for policy_noise in 0.1 0.3 0.5
do 
python enjoy.py --env-kwargs "$var_param" --algo ppo --env RoboschoolHalfCheetah-v1 -f logs/ --exp-id 1 --load-best --save_file 'offline_data/roboschoolhalfcheetah' --timesteps $timesteps -n 10000 \
--policy_noise $policy_noise
done 
done 


