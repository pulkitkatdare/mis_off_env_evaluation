

for real_policy in 0.6 0.4
do
for sim_policy in 0.1 0.3 0.5
do 
for algo_type in 'Beta-DICE' 
do
python train_oee_estimator_continous.py --real_policy $real_policy --sim_policy $sim_policy --index 0 --algo_type $algo_type --env 'RoboschoolHalfCheetah-v1'
done
done
done

