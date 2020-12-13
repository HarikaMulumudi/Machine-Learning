from keras.datasets.mnist import load_data
from matplotlib import pyplot

# NIST data loading
(trainX, trainy), (testX, testy) = load_data()
trainX = trainX[:5000]
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from numpy import vstack
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Flatten
from keras.layers import LeakyReLU
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.layers import Dropout
from matplotlib import pyplot

# Discriminator model
def discriminator(in_shape=(28,28,1)):
	m = Sequential()
	m.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	m.add(LeakyReLU(alpha=0.2))
	m.add(Dropout(0.4))
	m.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	m.add(LeakyReLU(alpha=0.2))
	m.add(Dropout(0.4))
	m.add(Flatten())
	m.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	m.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return m

# Generator model
def generator(latent_dim):
	m = Sequential()
	n = 128 * 7 * 7
	m.add(Dense(n, input_dim=latent_dim))
	m.add(LeakyReLU(alpha=0.2))
	m.add(Reshape((7, 7, 128)))
	m.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	m.add(LeakyReLU(alpha=0.2))
	m.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	m.add(LeakyReLU(alpha=0.2))
	m.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return m

# GAN Model
def define_gan(g, d):
	d.trainable = False
	m = Sequential()
	m.add(g)
	m.add(d)
	opt = Adam(lr=0.0002, beta_1=0.5)
	m.compile(loss='binary_crossentropy', optimizer=opt)
	return m

# Load the data
def load_samples():
	(train_X, _), (_, _) = load_data()
	Xdim = expand_dims(train_X, axis=-1)
	Xdim = Xdim.astype('float32')
	Xdim = Xdim / 255.0
	return Xdim

# choose samples
def generate_samples(dataset, n):
	ix = randint(0, dataset.shape[0], n)
	Xdim = dataset[ix]
	Ydim = ones((n, 1))
	return Xdim, Ydim

# latent space points generation
def generate_points(dim, n):
	xInput = randn(dim * n)
	xInput = xInput.reshape(n, dim)
	return xInput

# generate fake digits
def generate_fake(g, dim, n):
	xInput = generate_points(dim, n)
	Xdim = g.predict(xInput)
	Ydim = zeros((n, 1))
	return Xdim, Ydim

# saving the plots
def save(ex, epoch, n=200):
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(ex[i, :, :, 0], cmap='gray_r')
	pyplot.show()
	pyplot.close()

#Model Evaluation
def summarize_performance(epoch, g, d, data_set, dim, n=100):
	XReal, yReal = generate_samples(data_set, n)
	_, accReal = d.evaluate(XReal, yReal, verbose=0)
	xFake, yFake = generate_fake(g, dim, n)
	_, accFake = d.evaluate(xFake, yFake, verbose=0)
	print('>Score real images: %.0f%%, fake images: %.0f%%' % (acc_real*100, acc_fake*100))
	save(xFake, epoch)
	g.show()

# Training Phase
def train(g, d, gan, data_set, dim, n=200, nb=256):
	bat = int(dataset.shape[0] / nb)
	halfBatch = int(nb / 2)
	for i in range(n):
		for j in range(bat):
			XReal, yReal = generate_samples(data_set, halfBatch)
			XFake, yFake = generate_fake(g, dim, halfBatch)
			Xdim, Ydim = vstack((XReal, XFake)), vstack((yReal, yFake))
			dLoss, _ = d.train_on_batch(Xdim, Ydim)
			XGan = generate_points(dim, nb)
			yGan = ones((nb, 1))
			gLoss = gan.train_on_batch(XGan, yGan)
		print('>%d, %d/%d, d is %.3f, g is %.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		if (i+1) % 5 == 0:
			summarize_performance(i, g, d, data_set, dim)

dim = 100
d = discriminator()
g = generator(dim)
gan = define_gan(g, d)
dataset = load_samples()
train(g, d, gan, dataset, dim)

def printing() :
  xInput = generate_points(100, 100)
  trainX = g.predict(xInput)
  for i in range(100):
    pyplot.subplot(10, 10, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(trainX[i, :, :, 0], cmap='gray_r')
  pyplot.show()

print(g.summary(line_length=None, positions=None, print_fn=None))
print(d.summary(line_length=None, positions=None, print_fn=None))
print(gan.summary(line_length=None, positions=None, print_fn=None))

printing()