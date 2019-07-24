######################################################################
# 500 images 20x20-20-tanh_linear-0.05--1.0-5.0-40-0.5 targetd attack
#####################################################################
#sudo python3 generate_targeted_attack_metabatch.py -N=500 -seed=42 -metabatch_size=10

####################################################################
# 6x50 images 20x20-20-tanh_linear-0.05--1.0-5.0-120-0.5 targetd attack
#####################################################################
#sudo python3 generate_targeted_attack_metabatch.py -N=500 -seed=42 -metabatch_size=10
sudo python3 generate_targeted_attack_metabatch.py -N=50 -seed=42 -nb_iter=120 -metabatch_size=10
sudo python3 generate_targeted_attack_metabatch.py -N=50 -seed=189 -nb_iter=120 -metabatch_size=10
sudo python3 generate_targeted_attack_metabatch.py -N=50 -seed=182 -nb_iter=120 -metabatch_size=10
sudo python3 generate_targeted_attack_metabatch.py -N=50 -seed=154 -nb_iter=120 -metabatch_size=10
sudo python3 generate_targeted_attack_metabatch.py -N=50 -seed=157 -nb_iter=120 -metabatch_size=10
sudo python3 generate_targeted_attack_metabatch.py -N=50 -seed=127 -nb_iter=120 -metabatch_size=10
