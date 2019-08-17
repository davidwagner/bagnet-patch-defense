import tensorflow as tf
from torch import optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import bagnets.pytorch
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, SPSA, projected_optimization
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print(torch.cuda.get_device_name(0))


bagnet33 = bagnets.pytorch.bagnet33(pretrained=True, avg_pool=True).to(device)
bagnet33.eval()
model = bagnet33
print('loaded model')
# ImageNet validation set
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
imagenet_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(), 
                                         normalize])
imagenet_val = datasets.ImageNet("./", split='val', download=False,
                                     transform=imagenet_transform)

val_subset_indices = image_partition(42, 100)[0]
val_subset_loader = torch.utils.data.DataLoader(imagenet_val,
                                                batch_size=1,
                                                num_workers=4,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))

# We use tf for evaluation on adversarial data
sess = tf.Session()
x_op = tf.placeholder(tf.float32, shape=(None, 3, 224, 224,))
# Convert pytorch model to a tf_model and wrap it in cleverhans
tf_model_fn = convert_pytorch_model_to_tf(model)
cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
cleverhans_model.nb_classes = 1000

spsa_op = SPSA(cleverhans_model, sess=sess)
spsa_params = {'eps': 2.5,
             'clip_min': -2.3,
             'clip_max': 2.8, 
             'nb_iter': 40,
             'y': None}

for xs, ys in val_subset_loader:
    count += 1
    print(count)
    # Create an SPSA attack
    spsa_params['y'] = ys
    adv_x_op = spsa_op.generate(x_op, **spsa_params)
    #adv_preds_op = tf_model_fn(adv_x_op)
    adv_x = sess.run(adv_x_op, feed_dict={x_op: xs})
    print(adv_x.shape)
    break
