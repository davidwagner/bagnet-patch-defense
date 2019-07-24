#####################################################################
# 500 images 20x20-20-tanh_linear-0.05--1.0-5.0-40-0.5 targetd attack
#####################################################################
sudo python3 generate_targeted_attack_metabatch.py --N=500 --seed=42 --metabatch_size=10
sudo python3 generate_targeted_attack_metabatch.py --N=500 --seed=88 --metabatch_size=10

