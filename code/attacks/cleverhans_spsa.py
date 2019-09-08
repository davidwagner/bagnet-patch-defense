from keras import backend as K
from keras.layers import Layer, Input
from keras.models import Model
from keras.preprocessing import image as KImage
from keras.preprocessing import image as KImage
from cleverhans.attacks import SPSA
from cleverhans.model import CallableModelWrapper
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
from attacks.tensor import*

DEBUG = False
class ApplyStickerLayer(Layer):
    def __init__(self, img, size, loc):
        """Save IMG as base image. SIZE sticker will be pasted at LOC."""
        super(ApplyStickerLayer, self).__init__()
        
        # how much to shift the stick over
        self.shift = tf.constant(loc, name='shift')
        
        # preparing the base image, zero out the space where the sticker goes
        x1, y1 = loc
        x2, y2 = x1 + size[0], y1 + size[1]
        img[:, :, x1:x2, y1:y2] = 0 # shape = (1, 3, 224, 224)
        self.base_image = tf.constant(img, dtype=tf.float32, name='base-image')
        
        # store the shapes we need
        self.out_shape = (None,) + img.shape[1:]
        self.full_shape = tf.constant(img.shape[1:], dtype=tf.int64, name='full-shape')
        
    
    @debug_build(DEBUG)
    def build(self, _dummy_input_shape):
        pass
    
    @debug_call(DEBUG)
    def call(self, subimg):
        """Paste input sticker into base image."""
        # Get the shape of subimag (need the batch size in it later)
        batch_size = tf.slice(K.shape(subimg), [0], [1])
        batch_size = tf.dtypes.cast(batch_size, dtype=tf.int64)
        
        # Concat the batch size with the full shape
        full_shape_batch = tf.concat([batch_size, self.full_shape], 0) 
        
        # Expand sticker to full image
        idx = tf.where(tf.not_equal(subimg, 0))
        stick_on = tf.SparseTensor(idx, tf.gather_nd(subimg, idx), full_shape_batch)
        stick_on = tf.sparse.to_dense(stick_on)
        stick_on = tf.roll(stick_on, shift=self.shift, axis=(2, 3))
        
        return tf.add(stick_on, self.base_image)
    
    @debug_compute(DEBUG)
    def compute_output_shape(self, _dummy_input_shape):
        return self.out_shape

class CleverhansDataLoader:
    def __init__(self, images, labels):
        """ An iterator yielding one image and its label at a time from a batch of images
        Input:
        - images (numpy array): shape (batch_size, 3, 224, 224)
        - labels (numpy array): shape (batch_size, )
        """
        self.images = images
        self.labels = labels
        self.length = images.shape[0]
        self.count = -1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.count += 1
        if self.count < self.length:
            return (self.images[self.count][None].copy(), np.array(self.labels[self.count]))
        else:
            raise StopIteration
        
    def __len__(self):
        return self.images.shape[0]

def load_image(img_path, mean=np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), std=np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))):
    img = KImage.load_img(img_path, target_size=(224, 224))
    img = KImage.img_to_array(img).transpose([2, 0, 1])
    img /= 255
    img = (img - mean)/std
    return img

#####################################################################
# CleverHans SPSA Sticker attack
#####################################################################
def cleverhans_spsa(model, data_iter, attack_size):
    count = 0
    try: 
        image, label = next(data_iter)
        label = np.array([label])
    except:
        break
    print('Start attacking image {}'.format(count))
    for x in range():
        for y in range():
            print("location {}".format((x, y)))
            subimg = get_subimgs(image, (x, y), attack_size)
            #Build model
            tic = time.time()
            subimg_op = Input(shape=(3, attack_size[0], attack_size[1]))
            adv_img_op = ApplyStickerLayer(image, attack_size, (x, y))(subimg_op)
            wrapped_logits_op = model(adv_img_op)
            wrapped_model = Model(inputs=subimg_op, outputs=wrapped_logits_op)
            tac = time.time()
            print('{}s to build graph for attack'.format(tac - tic))
            wrapper = CallableModelWrapper(wrapped_model, "logits")
            wrapper.nb_classes = 1000
            attack = SPSA(wrapper, sess=keras.backend.get_session())
            spsa_params = {'eps': 2.5,
               'clip_min': -2.3,
               'clip_max': 2.7,
               'nb_iter': 40,
               'y': label.astype(np.int32)}
            print('Start attacking...')
            tic = time.time()
            adv = attack.generate_np(subimg, **spsa_params)
            tac = time.time()
            print("Attack Time: {}s".format(tac - tic))

            # Evaluate adversarial sticker
            adv_logits = wrapped_model.predict(adv)
            print("Adversarial image: top-5 prediction: {}, label: {}".format(np.argsort(adv_logits, axis=1)[:, -5:], label))
