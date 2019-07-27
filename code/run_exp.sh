####################################################################
# 50x6 images 20x20-20-tanh_linear-0.05--1.0-5.0-40-0.5 (Finished)
####################################################################
#sudo python3 generate_new_metabatch.py --seed=42 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=88 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=1234 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=666 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=777 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=999 --metabatch_size=10

####################################################################
# 500 images 20x20-20-tanh_linear-0.05--1.0-5.0-40-0.5 (Finished)
####################################################################
#sudo python3 generate_new_metabatch.py --N=500 --seed=42 --metabatch_size=10

####################################################################
# 500 images 20x20-20-sigmoid_linear-100000.0--2500000-5.0-40-0.5 (Finished)
####################################################################
#sudo python3 generate_new_metabatch.py --N=500 --seed=42 --clip_fn=sigmoid_linear --a=100000.0 --b=-2500000.0 --metabatch_size=10

#####################################################################
# 50x6 images 20x20-20-tanh_linear-0.05--1.0-5.0-120-0.5 (Finished)
#####################################################################
#sudo python3 generate_new_metabatch.py --seed=666 --nb_iter=120 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=1234 --nb_iter=120 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=42 --nb_iter=120 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=88 --nb_iter=120 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=777 --nb_iter=120 --metabatch_size=10
#sudo python3 generate_new_metabatch.py --seed=999 --nb_iter=120 --metabatch_size=10

#####################################################################
# 500 images canonical BagNet-9, -17 (Finished)
#####################################################################
#sudo python3 generate_generic_new_metabatch.py -N=500 -seed=42 -metabatch_size=20 -model=bagnet9
#sudo python3 generate_generic_new_metabatch.py -N=500 -seed=42 -metabatch_size=20 -model=bagnet17
#sudo python3 generate_generic_new_metabatch.py -N=500 -seed=88 -metabatch_size=10 -model=bagnet17
#sudo python3 generate_generic_new_metabatch.py -N=500 -seed=88 -metabatch_size=10 -model=bagnet9

######################################################################
# 500 images random initialization, nb_iter = 40, 80, 120 (In progress)
######################################################################
sudo python3 generate_new_metabatch.py -rand_init=True -N=500 -seed=42 -nb_iter=40 -stepsize=0.125 -metabatch_size=10 -model=bagnet33
sudo python3 generate_new_metabatch.py -rand_init=True -N=500 -seed=42 -nb_iter=80 -stepsize=0.0625 -metabatch_size=10 -model=bagnet33
sudo python3 generate_new_metabatch.py -rand_init=True -N=500 -seed=42 -nb_iter=120 -stepsize=0.04 -metabatch_size=10 -model=bagnet33

######################################################################
# 500 images random initialization, several step sizes (Planed)
######################################################################
#sudo python3 generate_new_metabatch.py -rand_init=True -N=500 -seed=42 -nb_iter=40 -stepsize=0.225 -metabatch_size=10 -model=bagnet33
#sudo python3 generate_new_metabatch.py -rand_init=True -N=500 -seed=42 -nb_iter=40 -stepsize=0.325 -metabatch_size=10 -model=bagnet33
#sudo python3 generate_new_metabatch.py -rand_init=True -N=500 -seed=42 -nb_iter=40 -stepsize=0.425 -metabatch_size=10 -model=bagnet33
#sudo python3 generate_new_metabatch.py -rand_init=True -N=500 -seed=42 -nb_iter=40 -stepsize=0.5 -metabatch_size=10 -model=bagnet33
#sudo python3 generate_new_metabatch.py -rand_init=True -N=500 -seed=42 -nb_iter=40 -stepsize=0.5 -metabatch_size=10 -model=bagnet33
