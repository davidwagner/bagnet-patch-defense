#python3 generated_generic_new_metabatch.py --model=alexnet

#python3 generated_generic_new_metabatch.py --model=vgg16

#python3 generated_generic_new_metabatch.py --model=resnet18

#python3 generated_generic_new_metabatch.py --model=resnet34
#python3 generated_generic_new_metabatch.py --model=bagnet9
#python3 generated_generic_new_metabatch.py --model=bagnet17
#python3 generated_generic_new_metabatch.py --model=bagnet3
########################################################################
# 2017-7-29 Baseline (Finished)
########################################################################
#python3 generate_generic_new_metabatch.py --model=resnet50 --nb_iter=10 --stepsize=0.5
#python3 generate_generic_new_metabatch.py --model=densenet --nb_iter=10 --stepsize=0.5
#python3 generate_generic_new_metabatch.py --model=inception --nb_iter=10 --stepsize=0.5

#python3 generate_generic_new_metabatch.py --model=resnet50 --nb_iter=20 --stepsize=0.25
#python3 generate_generic_new_metabatch.py --model=densenet --nb_iter=20 --stepsize=0.25
#python3 generate_generic_new_metabatch.py --model=inception --nb_iter=20 --stepsize=0.25

########################################################################
# 2017-7-30 ~ 31 Baseline (In progess)
########################################################################
#python3 generate_generic_new_metabatch.py -model=resnet101 -nb_iter=10 -stepsize=0.5
#python3 generate_generic_new_metabatch.py -model=resnet152 -nb_iter=10 -stepsize=0.5

#python3 generate_generic_new_metabatch.py -model=resnet101 -nb_iter=20 -stepsize=0.25
#python3 generate_generic_new_metabatch.py -model=resnet152 -nb_iter=20 -stepsize=0.25

#python3 generate_generic_new_metabatch.py -model=resnet50 -nb_iter=40 -stepsize=0.125
#python3 generate_generic_new_metabatch.py -model=resnet101 -nb_iter=40 -stepsize=0.125
#python3 generate_generic_new_metabatch.py -model=resnet152 -nb_iter=40 -stepsize=0.125
#python3 generate_generic_new_metabatch.py -model=densenet -nb_iter=40 -stepsize=0.125
#python3 generate_generic_new_metabatch.py -model=inception -nb_iter=40 -stepsize=0.125

#python3 generate_generic_new_metabatch.py -model=resnet50 -nb_iter=80 -stepsize=0.0625
#python3 generate_generic_new_metabatch.py -model=resnet101 -nb_iter=80 -stepsize=0.0625 
#python3 generate_generic_new_metabatch.py -model=resnet152 -nb_iter=80 -stepsize=0.0625
#python3 generate_generic_new_metabatch.py -model=densenet -nb_iter=80 -stepsize=0.0625
#python3 generate_generic_new_metabatch.py -model=inception -nb_iter=80 -stepsize=0.0625

#python3 generate_generic_new_metabatch.py -model=bagnet9 -nb_iter=10 -stepsize=0.5
#python3 generate_generic_new_metabatch.py -model=bagnet17 -nb_iter=10 -stepsize=0.5
#python3 generate_generic_new_metabatch.py -model=bagnet33 -nb_iter=10 -stepsize=0.5
#
#python3 generate_generic_new_metabatch.py -model=bagnet9 -nb_iter=20 -stepsize=0.25
#python3 generate_generic_new_metabatch.py -model=bagnet17 -nb_iter=20 -stepsize=0.25
#python3 generate_generic_new_metabatch.py -model=bagnet33 -nb_iter=20 -stepsize=0.25
#
#python3 generate_generic_new_metabatch.py -model=bagnet9 -nb_iter=40 -stepsize=0.125
#python3 generate_generic_new_metabatch.py -model=bagnet17 -nb_iter=40 -stepsize=0.125
#python3 generate_generic_new_metabatch.py -model=bagnet33 -nb_iter=40 -stepsize=0.125
#
#python3 generate_generic_new_metabatch.py -model=bagnet9 -nb_iter=80 -stepsize=0.0625
################################################################################### Checkpoint
#python3 generate_generic_new_metabatch.py -model=bagnet9 -nb_iter=40 -stepsize=0.5
#python3 generate_generic_new_metabatch.py -model=bagnet17 -nb_iter=40 -stepsize=0.5
#python3 generate_generic_new_metabatch.py -model=bagnet33 -nb_iter=40 -stepsize=0.5
#python3 generate_generic_new_metabatch.py -model=bagnet9 -nb_iter=80 -stepsize=0.0625
#python3 generate_generic_new_metabatch.py -model=bagnet17 -nb_iter=80 -stepsize=0.0625
#python3 generate_generic_new_metabatch.py -model=bagnet33 -nb_iter=80 -stepsize=0.0625
###################################################################################

#################################################################################
# Step size testing (Unfinished)
#################################################################################
#python3 generate_generic_new_metabatch.py -model=resnet50 -nb_iter=10 -stepsize=2.
#python3 generate_generic_new_metabatch.py -model=resnet50 -nb_iter=20 -stepsize=1.
#python3 generate_generic_new_metabatch.py -model=bagnet9 -nb_iter=10 -stepsize=2.
#python3 generate_generic_new_metabatch.py -model=bagnet9 -nb_iter=20 -stepsize=1.
#python3 generate_generic_new_metabatch.py -model=bagnet17 -nb_iter=10 -stepsize=2.
#python3 generate_generic_new_metabatch.py -model=bagnet9 -nb_iter=80 -stepsize=0.25
#python3 generate_generic_new_metabatch.py -model=bagnet17 -nb_iter=80 -stepsize=0.25

# resnet50 ratio=20
#python3 generate_generic_new_metabatch.py -model=resnet50 -nb_iter=40 -stepsize=0.5
#python3 generate_generic_new_metabatch.py -model=resnet50 -nb_iter=80 -stepsize=0.25

# vanilla bagnet33 ratio=20
#python3 generate_generic_new_metabatch.py -model=bagnet33 -nb_iter=10 -stepsize=2.
python3 generate_generic_new_metabatch.py -model=bagnet33 -nb_iter=20 -stepsize=1.
python3 generate_generic_new_metabatch.py -model=bagnet33 -nb_iter=80 -stepsize=0.25
# vanilla bagnet33 ratio=5
python3 generate_generic_new_metabatch.py -model=bagnet33 -nb_iter=80 -stepsize=0.0625

# vanilla bagnet17 ratio=20
python3 generate_generic_new_metabatch.py -model=bagnet17 -nb_iter=20 -stepsize=1.
# vanilla bagnet17 ratio=5
python3 generate_generic_new_metabatch.py -model=bagnet17 -nb_iter=80 -stepsize=0.0625
