###############################################################
# 500 images scheduled param 20x20-20-binarize-25-5.0-200-0.025
###############################################################
sudo python3 generate_scheduled_param_metabatch.py -model=bagnet33 -N=500 -seed=42 -nb_iter=200 -stepsize=0.025 -rand_init=True
