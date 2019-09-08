sudo python3 foolbox_attack.py -N=500 -model=bagnet33 -clip_fn=tanh_linear -attack_alg=AdamRandomPGD -nb_iter=40 -stepsize=0.1 -get_robust=True
