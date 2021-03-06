import torch
from keras import backend as K
from keras.layers import Layer, Input
from keras.models import Model
import bagnets.keras
from attacks.cleverhans_spsa import*
from absl import app, flags
import os
 
FLAGS = flags.FLAGS

def main(argv):
    """
    FLAGS.output_root/
        [NAME]/
            [NAME].mtb
            [NAME].log
            dataset/
    """
    ###################################
    # Model and data preparation
    ###################################

    # GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        print(device)

    # ImageNet validation set
    # This is where you put your ImageNet validation data.
    val_path = '/mnt/data/imagenet/val'
    mean=np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std=np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
	
    count, max_imgs = 0, 1000
    paths = []
    for i, imagenet_class in enumerate(sorted(os.listdir(val_path))):
        if i < max_imgs:
            path = os.path.join(val_path, imagenet_class)
            paths.append(os.path.join(path,os.listdir(path)[0])) # just take the first file from each directory
        else:
            break
    bs = max_imgs
    attack_size = (20, 20)
    batch_imgs = np.concatenate([load_image(paths[i])[None] for i in range(bs)], axis=0)
	#data_iter = iter(CleverhansDataLoader(batch_imgs, np.arange(bs)))

    # load pretrained model
    bagnet33_keras = bagnets.keras.bagnet33()
    #bagnet17_keras = bagnets.keras.bagnet17()
    #bagnet9_keras = bagnets.keras.bagnet9()
    model = bagnet33_keras

    for layer in model.layers:
        layer.trainable = False

    #####################################
    # Start attacking
    #####################################
    print("Start attacking...")
    data_iter = iter(CleverhansDataLoader(batch_imgs, np.arange(bs)))
    num_correct = 0
    while True:
        try:
            image, label = next(data_iter)
        except StopIteration:
            break
        logits = model.predict(image)
        topk = np.argsort(logits, axis=1)[:, -5:]  
        is_correct = label in topk[0]
        if is_correct:
            num_correct += 1
        print("Top-5 prediction: {}, label: {}".format(topk, label))
        print("Is correct: {}".format(is_correct))
    print("Clean accuray: {}".format(num_correct/bs))

"""
    data_iter = iter(CleverhansDataLoader(batch_imgs, np.arange(bs)))
    tic = time.time()
    succ_prob = cleverhans_spsa(model, data_iter, attack_size)
    tac = time.time()
    print(setting_info)
    print("Success probability: {}, Time: {:.3f}s or {:.3f}hr(s)".format(succ_prob, tac - tic, (tac-tic)/3600))
"""
if __name__ == "__main__":
    app.run(main)

