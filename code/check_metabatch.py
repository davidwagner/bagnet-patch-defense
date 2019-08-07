from bagnets.clipping import*
from bagnets.security import*
from absl import app, flags
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string('name', None, 'metabatch name')
flags.DEFINE_string('output_root', '/mnt/data/results/advertorch_results', 'directory for storing results')

def main(argv):
    METABATCH_PATH = os.path.join(FLAGS.output_root, FLAGS.name, FLAGS.name+'.mtb')
    with open(METABATCH_PATH, 'rb') as file:
        metabatch = pickle.load(file)
    print("Accuracy before attack of {}: {}".format(FLAGS.name, metabatch.clean_acc))
    print("Successful defense probability of {}: {} \n".format(FLAGS.name, metabatch.get_succ_prob()))
    
if __name__ == "__main__":
    app.run(main)
