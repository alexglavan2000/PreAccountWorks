# Write your assignment here

datasets, datasets_info = tfds.load(name='even_mnist')

def _preprocess(sample):
  image = tf.cast(sample['image'], tf.float32) / 255. 
  image = image < tf.random.uniform(tf.shape(image))   
  return image, image

train_dataset = (datasets['train'].map(_preprocess).batch(256).prefetch(tf.data.AUTOTUNE).shuffle(int(10e3)))
eval_dataset = (datasets['test'].map(_preprocess).batch(256).prefetch(tf.data.AUTOTUNE))

encoder = tfk.Sequential([tfkl.InputLayer(input_shape=input_shape),tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),tfkl.Conv2D(base_depth, 5, strides=1,padding='same', activation=tf.nn.leaky_relu),tfkl.Conv2D(base_depth, 5, strides=2,padding='same', activation=tf.nn.leaky_relu),tfkl.Conv2D(2 * base_depth, 5, strides=1,padding='same', activation=tf.nn.leaky_relu),tfkl.Conv2D(2 * base_depth, 5, strides=2,padding='same', activation=tf.nn.leaky_relu),tfkl.Conv2D(4 * encoded_size, 7, strides=1,padding='valid', activation=tf.nn.leaky_relu),tfkl.Flatten(),tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),activation=None),tfpl.MultivariateNormalTriL(encoded_size,activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),])

decoder = tfk.Sequential([tfkl.InputLayer(input_shape=[encoded_size]),tfkl.Reshape([1, 1, encoded_size]),tfkl.Conv2DTranspose(2 * base_depth, 7, strides=1,padding='valid', activation=tf.nn.leaky_relu),tfkl.Conv2DTranspose(2 * base_depth, 5, strides=1,padding='same', activation=tf.nn.leaky_relu),tfkl.Conv2DTranspose(2 * base_depth, 5, strides=2,padding='same', activation=tf.nn.leaky_relu),tfkl.Conv2DTranspose(base_depth, 5, strides=1,padding='same', activation=tf.nn.leaky_relu),tfkl.Conv2DTranspose(base_depth, 5, strides=2,padding='same', activation=tf.nn.leaky_relu),tfkl.Conv2DTranspose(base_depth, 5, strides=1,padding='same', activation=tf.nn.leaky_relu),tfkl.Conv2D(filters=1, kernel_size=5, strides=1,padding='same', activation=None),tfkl.Flatten(),tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),])

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),loss=negloglik)

_ = vae.fit(train_dataset,epochs=15,validation_data=eval_dataset)

# We'll just examine ten random digits.
x = next(iter(eval_dataset))[0][:10]
xhat = vae(x)
assert isinstance(xhat, tfd.Distribution)

print('Originals:')
display_imgs(x)

print('Decoded Random Samples:')
display_imgs(xhat.sample())

print('Decoded Modes:')
display_imgs(xhat.mode())

print('Decoded Means:')
display_imgs(xhat.mean())

# Now, let's generate ten never-before-seen digits.
z = prior.sample(10)
xtilde = decoder(z)
assert isinstance(xtilde, tfd.Distribution)

print('Randomly Generated Samples:')
display_imgs(xtilde.sample())

print('Randomly Generated Modes:')
display_imgs(xtilde.mode())

print('Randomly Generated Means:')
display_imgs(xtilde.mean())
