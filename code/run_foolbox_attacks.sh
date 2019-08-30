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
# 2019-8-9~8-12 AdamRandomPGD v.s. Vanilla PGD (Finished)
#################################################################
#sudo python3 foolbox_attack.py -N=500 -model=bagnet9 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.025

#sudo python3 foolbox_attack.py -N=500 -model=bagnet9 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1

#sudo python3 foolbox_attack.py -N=500 -model=bagnet9 -clip_fn=None -attack_alg=PGD -nb_iter=40 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results

#sudo python3 foolbox_attack.py -N=500 -model=resnet50 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results

#sudo python3 foolbox_attack.py -N=500 -model=densenet -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results

#sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -clip_fn=tanh_linear -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 

#sudo python3 foolbox_attack.py -N=500 -model=resnet101 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results

##########################################################
# 2019-8-14 ~ 15 Targeted attack (average case) 
# AdamRandomPGD + CW objective
##########################################################
#testing
#sudo python3 foolbox_attack.py -N=5 -model=bagnet9 -clip_fn=None -attack_alg=AdamRandomPGD -targeted=False -nb_iter=40 -stepsize=0.1 
#sudo python3 foolbox_attack.py -N=5 -model=bagnet9 -clip_fn=None -attack_alg=AdamRandomPGD -targeted=True -nb_iter=40 -stepsize=0.1 
#sudo python3 foolbox_attack.py -N=5 -model=bagnet9 -clip_fn=tanh_linear -attack_alg=AdamRandomPGD -targeted=True -nb_iter=40 -stepsize=0.1 
#sudo python3 foolbox_attack.py -N=5 -model=bagnet9 -clip_fn=tanh_linear -attack_alg=AdamRandomPGD -targeted=False -nb_iter=40 -stepsize=0.1 

#sudo python3 foolbox_attack.py -N=500 -model=bagnet9 -clip_fn=None -attack_alg=AdamRandomPGD -targeted=True -nb_iter=40 -stepsize=0.1 
#sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -clip_fn=tanh_linear -attack_alg=AdamRandomPGD -targeted=True -nb_iter=40 -stepsize=0.1 
#sudo python3 foolbox_attack.py -N=500 -model=resnet50 -clip_fn=None -attack_alg=AdamRandomPGD -targeted=True -nb_iter=40 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results
#sudo python3 foolbox_attack.py -N=500 -model=densenet -clip_fn=None -attack_alg=AdamRandomPGD -targeted=True -nb_iter=40 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results

#########################################################
# 2019-8-17~ AdamRandomPGD untargeted iter=80
#########################################################
#sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -clip_fn=tanh_linear -attack_alg=AdamRandomPGD -nb_iter=80 -stepsize=0.1 
#sudo python3 foolbox_attack.py -N=500 -model=resnet50 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=80 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results
#sudo python3 foolbox_attack.py -N=500 -model=densenet -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=80 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results

#################################################################
# 2019-8-17~ AdamRandomPGD untargeted (non-representative models)
#################################################################
#sudo python3 foolbox_attack.py -N=500 -model=bagnet17 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1
#sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1
#sudo python3 foolbox_attack.py -N=500 -model=bagnet17 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=80 -stepsize=0.1
#sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=80 -stepsize=0.1

###############################################################
# 2019-8-29 Clipped BagNet-17 (Planed)
###############################################################
#sudo python3 foolbox_attack.py -N=500 -model=bagnet17 -clip_fn=tanh_linear -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 

################################################################
# 2019-8-29 critical region sticker attack
################################################################
sudo python3 foolbox_attack.py -N=500 -stride=60 -model=bagnet9 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 
sudo python3 foolbox_attack.py -N=500 -stride=60 -model=bagnet33 -clip_fn=tanh_linear -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 
#sudo python3 foolbox_attack.py -N=500 -stride=60 -model=densenet -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results
#sudo python3 foolbox_attack.py -N=500 -stride=60 -model=resnet50 -clip_fn=None -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 -data_path=/home/zhanyuan/data/imagenet -output_root=/home/zhanyuan/data/results/foolbox_results
