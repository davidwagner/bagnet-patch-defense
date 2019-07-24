#####################################################################
# 500 images canonical BagNet-9, -17 (Finished)
#####################################################################
sudo python3 clipping_params_searching.py -N=5000 -seed=42 -model=bagnet9 -clip_fn=tanh_linear -param=a param_list="[i*0.01 for i in range(101)]" -fixed_param=0. -data_path=/home/yuanbenson1772/data/imagenet -output_root=/home/yuanbenson1772/data/clipping_params_searching/
