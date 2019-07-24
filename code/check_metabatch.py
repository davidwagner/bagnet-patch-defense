from bagnets.clipping import*
from bagnets.security import*
from absl import app, flags
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_boolean('is_targeted', False, 'whether the attack is targeted')
flags.DEFINE_string('case', None, 'case of targeted attack')
flags.DEFINE_boolean('is_canonical', False, 'whether the model is canonical (v.s. with clipping)')
flags.DEFINE_integer('N', 50, 'number of images')
flags.DEFINE_integer('seed', 42, 'random seed for sampling images from ImageNet')
flags.DEFINE_multi_integer('attack_size', [20, 20], 'size of sticker')
flags.DEFINE_integer('stride', 20, 'stride of sticker')
flags.DEFINE_string('clip_fn', 'tanh_linear', 'clipping function')
flags.DEFINE_float('a', 0.05, 'clipping parameter A')
flags.DEFINE_float('b', -1, 'clipping parameter B')
flags.DEFINE_float('eps', 5., 'range of perturbation')
flags.DEFINE_integer('nb_iter', 40, 'number of iterations for PGD')
flags.DEFINE_float('stepsize', 0.5, 'stepsize of PGD')
flags.DEFINE_string('model', 'bagnet33', 'model that generates metabatch')
flags.DEFINE_string('output_root', '/mnt/data/results/', 'directory for storing results')

def main(argv):
    if FLAGS.is_targeted:
        NAME = 'targeted_{}_{}-{}-{}-{}-{}x{}-{}-{}-{}-{}-{}-{}-{}'.format(FLAGS.is_targeted, FLAGS.case, FLAGS.model, FLAGS.N, FLAGS.seed, FLAGS.attack_size[0], FLAGS.attack_size[1], FLAGS.stride, FLAGS.clip_fn, FLAGS.a, FLAGS.b, FLAGS.eps, FLAGS.nb_iter, FLAGS.stepsize)
    elif FLAGS.is_canonical:
        NAME = '{}-{}-{}x{}-{}-{}-{}-{}-{}'.format(FLAGS.N, FLAGS.seed, FLAGS.attack_size[0], FLAGS.attack_size[1], FLAGS.stride, FLAGS.model, FLAGS.eps, FLAGS.nb_iter, FLAGS.stepsize)
    else:
        NAME = '{}-{}-{}-{}x{}-{}-{}-{}-{}-{}-{}-{}'.format(FLAGS.model, FLAGS.N, FLAGS.seed, FLAGS.attack_size[0], FLAGS.attack_size[1], FLAGS.stride, FLAGS.clip_fn, FLAGS.a, FLAGS.b, FLAGS.eps, FLAGS.nb_iter, FLAGS.stepsize)
    OUTPUT_PATH = os.path.join(FLAGS.output_root, NAME)
    METABATCH_PATH = os.path.join(OUTPUT_PATH, NAME+'.mtb')
    with open(METABATCH_PATH, 'rb') as file:
        metabatch = pickle.load(file)
    print("Accuracy before attack of {}: {}".format(NAME, metabatch.clean_acc))
    print("Successful defense probability of {}: {} \n".format(NAME, metabatch.get_succ_prob()))
    
if __name__ == "__main__":
    app.run(main)
