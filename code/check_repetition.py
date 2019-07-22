import numpy as np
from absl import app, flags
import os
import logging

FLAGS = flags.FLAGS
flags.DEFINE_integer('N', 50, 'number of samples')
flags.DEFINE_multi_integer('seed_list', [42, 88, 1234, 666, 777, 999], 'list of random seeds')
flags.DEFINE_string('log_root', '/mnt/data/results/', 'directory for storing results')

def main(argv):
    NAME = "{}".format(FLAGS.seed_list)
    LOG_PATH = os.path.join(FLAGS.log_root, NAME+'.log')
    print("log to {}".format(LOG_PATH))

    logger = logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
    total = []
    for seed in FLAGS.seed_list:
        np.random.seed(seed)
        sub = np.random.choice(np.arange(50000), size=FLAGS.N, replace=False)
        logging.info("seed: {} \n indices: {}".format(seed, sub))
        total.append(sub[:])
    total = np.array(total).flatten()
    unique, count = np.unique(total, return_counts=True)
    has_rep = np.any([i != 1 for i in count])
    logging.info("has repetition: {}".format(has_rep))
    print(has_rep)
    return has_rep


if __name__ == "__name__":
    app.run(main)
