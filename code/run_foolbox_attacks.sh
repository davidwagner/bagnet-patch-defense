####################################################################
# Foolbox attack testing
####################################################################
sudo python3 foolbox_attack.py -N=20 -model=bagnet33 -clip_fn=None
sudo python3 foolbox_attack.py -N=20 -model=bagnet33 -clip_fn=tanh_linear
