####################################################################
# Foolbox attack testing (Finished)
####################################################################
#sudo python3 foolbox_attack.py -N=20 -model=bagnet33 -clip_fn=None
#sudo python3 foolbox_attack.py -N=20 -model=bagnet33 -clip_fn=tanh_linear

###################################################################
# Foolbox parameter searching 
###################################################################
#sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -nb_iter=40 -stepsize=0.5
#sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -nb_iter=40 -stepsize=0.25
#sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -nb_iter=20 -stepsize=0.05 # 1/20
#sudo python3 foolbox_attack.py -N=500 -model=bagnet9 -clip_fn=None  -nb_iter=40 -stepsize=0.025
#sudo python3 foolbox_attack.py -N=500 -model=bagnet17 -clip_fn=None -nb_iter=40 -stepsize=0.025

#sudo python3 foolbox_attack.py -N=500 -model=bagnet9 -nb_iter=40 -stepsize=0.025
#sudo python3 foolbox_attack.py -N=500 -model=bagnet17 -nb_iter=40 -stepsize=0.025
#sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -nb_iter=40 -stepsize=0.025

#################################################################
# AdamRandomPGD v.s. Vanilla PGD
#################################################################
#sudo python3 foolbox_attack.py -N=500 -model=bagnet9 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.025
#sudo python3 foolbox_attack.py -N=500 -model=bagnet9 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1
sudo python3 foolbox_attack.py -N=500 -model=bagnet9 -clip_fn=None -attack_alg=PGD -nb_iter=40 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results
