import numpy as np
import tensorflow_datasets as tds
import tensorflow_probability as tfp
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import matplotlib.pyplot as plt

tk = tf.keras
tkl = tf.keras.layers
tpl = tfp.layers
td = tfp.distributions

datasets, datasets_info = tds.load(name='mnist',
                                    with_info=True,
                                    as_supervised=False)

def _preprocess(sample):
  image = tf.cast(sample['image'], tf.float32) / 255.  # Scale to unit interval.
  image = image < tf.random.uniform(tf.shape(image))   # Randomly binarize.
  return image, image

train_dataset = (datasets['train']
                 .map(_preprocess)
                 .batch(256)
                 .prefetch(tf.data.experimental.AUTOTUNE)
                 .shuffle(int(10e3)))
eval_dataset = (datasets['test']
                .map(_preprocess)
                .batch(256)
                .prefetch(tf.data.experimental.AUTOTUNE))

input_shape = datasets_info.features['image'].shape
encoded_size = 16
base_depth = 32

prior = td.Independent(td.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)

encoder = tk.Sequential([
    tkl.InputLayer(input_shape=input_shape),
    tkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
    tkl.Conv2D(base_depth, 5, strides=1,
                padding='same', activation=tf.nn.leaky_relu),
    tkl.Conv2D(base_depth, 5, strides=2,
                padding='same', activation=tf.nn.leaky_relu),
    tkl.Conv2D(2 * base_depth, 5, strides=1,
                padding='same', activation=tf.nn.leaky_relu),
    tkl.Conv2D(2 * base_depth, 5, strides=2,
                padding='same', activation=tf.nn.leaky_relu),
    tkl.Conv2D(4 * encoded_size, 7, strides=1,
                padding='valid', activation=tf.nn.leaky_relu),
    tkl.Flatten(),
    tkl.Dense(tpl.MultivariateNormalTriL.params_size(encoded_size),
               activation=None),
    tpl.MultivariateNormalTriL(
        encoded_size,
        activity_regularizer=tpl.KLDivergenceRegularizer(prior)),
])

decoder = tk.Sequential([
    tkl.InputLayer(input_shape=[encoded_size]),
    tkl.Reshape([1, 1, encoded_size]),
    tkl.Conv2DTranspose(2 * base_depth, 7, strides=1,
                         padding='valid', activation=tf.nn.leaky_relu),
    tkl.Conv2DTranspose(2 * base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tkl.Conv2DTranspose(2 * base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
    tkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tkl.Conv2DTranspose(base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
    tkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tkl.Conv2D(filters=1, kernel_size=5, strides=1,
                padding='same', activation=None),
    tkl.Flatten(),
    tpl.IndependentBernoulli(input_shape, td.Bernoulli.logits),
])

vae = tk.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]))
print(vae.summary(line_length=None, positions=None, print_fn=None))
print(encoder.summary(line_length=None, positions=None, print_fn=None))
print(decoder.summary(line_length=None, positions=None, print_fn=None))

"""#### Do inference."""

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
            loss=negloglik)

_ = vae.fit(train_dataset,
            epochs=200,
            validation_data=eval_dataset)
x = next(iter(eval_dataset))[0][:100]
xhat = vae(x)
assert isinstance(xhat, td.Distribution)



def display_imgs(x, y=None):
  if not isinstance(x, (np.ndarray, np.generic)):
    x = np.array(x)
  plt.ioff()
  n = x.shape[0]
  fig, axs = plt.subplots(10, 10, figsize=(10, 10))
  if y is not None:
    fig.suptitle(np.argmax(y, axis=1))
  for i in range(n):
    axs.flat[i].imshow(x[i].squeeze(), interpolation='none', cmap='gray')
    axs.flat[i].axis('off')
  plt.show()
  plt.close()
  plt.ion()


print('Originals:')
display_imgs(x)

print('Decoded Random Samples:')
display_imgs(xhat.sample())

print('Decoded Modes:')
display_imgs(xhat.mode())

print('Decoded Means:')
display_imgs(xhat.mean())

# Now, let's generate ten never-before-seen digits.
z = prior.sample(100)
xtilde = decoder(z)
assert isinstance(xtilde, td.Distribution)

print('Randomly Generated Samples:')
display_imgs(xtilde.sample())

print('Randomly Generated Modes:')
display_imgs(xtilde.mode())

print('Randomly Generated Means:')
display_imgs(xtilde.mean())