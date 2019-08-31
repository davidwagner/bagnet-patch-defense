from util.tensor import*

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
