
# https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
# https://github.com/wiseodd/generative-models/blob/master/VAE/vanilla_vae/vae_tensorflow.py
# https://danijar.com/building-variational-auto-encoders-in-tensorflow/

import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
tfd = tf.contrib.distributions
from .utils import InputError, is_positive_integer, ceil
from qml import Compound
from sklearn.manifold import *
from sklearn.decomposition import *
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.discriminant_analysis import *
from sklearn.calibration import CalibratedClassifierCV

class VAE(object):
    """
    Dimensionality reduction with variational autoencoders.
    The code is slightly modified from 
    https://danijar.com/building-variational-auto-encoders-in-tensorflow/

    """


    def __init__(self, dimensions = 2, layer_sizes = [200,200], learning_rate = 0.001,
            activation_function = tf.nn.relu, n_samples = 1, n_iterations = 20, batch_size = 100,
            filenames = []):
        self.dimensions = dimensions
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self._generate_compounds(filenames)

    def _get_batch_size(self):
        """
        Determines the actual batch size.
        If the batch size is larger than the number of samples, it is truncated and a warning
        is printed.

        Furthermore the returned batch size will be slightly modified from the user input if
        the last batch would be tiny compared to the rest.

        :return: Batch size
        :rtype: int
        """

        if self.batch_size > self.n_samples:
            print("Warning: batch_size larger than sample size. It is going to be clipped")
            return min(self.n_samples, self.batch_size)
        else:
            batch_size = self.batch_size

        # see if the batch size can be modified slightly to make sure the last batch is similar in size
        # to the rest of the batches
        # This is always less that the requested batch size, so no memory issues should arise
        better_batch_size = ceil(self.n_samples, ceil(self.n_samples, batch_size))

        return better_batch_size

    def _make_encoder(self, data):
        """
        ReLU activation on hidden layers is used and softplus
        to force scale to be positive.
        """
        x = tf.layers.flatten(data)
        for layer_size in self.layer_sizes:
            x = tf.layers.dense(x, layer_size, self.activation_function)

        loc = tf.layers.dense(x, self.dimensions)
        scale = tf.layers.dense(x, self.dimensions, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale), loc

    def _make_prior(self):
        """
        This could in principle be changed to a different distribution
        if we want a more uniform distribution or something
        distributed as a square instead of a circle (in 2D)
        """
        loc = tf.zeros(self.dimensions)
        scale = tf.ones(self.dimensions)
        return tfd.MultivariateNormalDiag(loc, scale)

    def _make_decoder(self, z):
        """
        Decoder uses the same layer_sizes as the encoder,
        but this is not necessarily optimal.
        ReLU is applied on the output layer since all input
        features are assumed positive, but exp could be used
        as well.
        Returns a gamma distribution since input is positive or zero.
        A mixture might be better since the input is sparse.
        Alternatively a beta-like distribution could be used if the max/min of the output
        was stored or the input was normalised.

        """

        for layer_size in self.layer_sizes:
            z = tf.layers.dense(z, layer_size, self.activation_function)

        #x = tf.layers.dense(x, self.n_features, tf.nn.relu)
        alpha = tf.layers.dense(z, self.n_features, tf.nn.softplus)
        beta = tf.layers.dense(z, self.n_features, tf.nn.softplus)
        return tfd.Independent(tfd.Gamma(alpha, beta), 1)

    def _generate_compounds(self, filenames):
        """
        Creates QML compounds.

        :param filenames: path of xyz-files
        :type filenames: list
        """

        self.compounds = np.empty(len(filenames), dtype=object)
        for i, filename in enumerate(filenames):
            self.compounds[i] = Compound(filename)


    def _get_slatm(self):
        mbtypes = self._get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
        x = np.empty(len(self.compounds), dtype=object)
        coords = []
        for i, mol in enumerate(self.compounds):
            #mol.generate_slatm(mbtypes, local = False)
            mol.generate_coulomb_matrix(size = 20, sorting = "unsorted")
            x[i] = mol.representation
            coords.append(mol.coordinates)
        x = np.asarray(list(x), dtype=float)
        coords = np.asarray(coords)

        return x, coords

    def _get_slatm_mbtypes(self, arr):
        from qml.representations import get_slatm_mbtypes
        return get_slatm_mbtypes(arr)

    def fit(self):
        x, coords = self._get_slatm()

        d1 = np.sum((coords[:,0] - coords[:,3])**2, axis = 1)
        d2 = np.sum((coords[:,4] - coords[:,3])**2, axis = 1)
        print(d1.shape)


        n = len(self.compounds)
        idx = np.random.randint(0, n, size = 100)
        range_ = np.arange(n)

        # Remove constant features
        x = x[:, x.std(0) > 1e-6]
        x_test = x[idx]
        x_train = x[~np.isin(range_,idx)]

        x0 = x[:1000]
        x1 = x[-1000:]
        xc = np.concatenate([x0,x1], axis=0)
        y = np.zeros(2000, dtype=int)
        y[-1000:] = 1

        import matplotlib.pyplot as plt



        """
        Run a classifier and use probabilities.
        Alternatively run regressor and modify hyperparams
        to make the transition from one state as sharp as needed.
        """
        #mod = LogisticRegression()
        #dist = {"C": 10**np.linspace(-1, 2, 1000),
        #        "penalty": ["l1", "l2"]}
        #cv_gen = RandomizedSearchCV(mod, dist, cv = 5, verbose = 1, n_iter = 20, refit=False)
        #cv_gen.fit(xc,y)
        #print(cv_gen.best_params_, cv_gen.best_score_)
        #mod.set_params(**cv_gen.best_params_)
        ##mod = CalibratedClassifierCV(mod, cv=3, method = "sigmoid")
        #mod.fit(xc,y)
        #pred = mod.predict_proba(x[1000:-1000])

        #plt.scatter(range(n-2000), pred[:,0])
        #plt.show()

        """
        Do dimensionality reduction
        """

        #mod = MDS(2)
        #y = mod.fit_transform(x)

        #plt.scatter(y[:1000,0], y[:1000,1], alpha = 0.5)
        #plt.scatter(y[-1000:,0], y[-1000:,1], alpha = 0.5)
        #plt.scatter(y[1000:-1000,0], y[1000:-1000,1], alpha = 0.5)
        #plt.show()

        #for i in range(y.shape[1]):
        #    plt.scatter(range(n), y[:,i])
        #    plt.show()

        plt.scatter(d1[:1000], d2[:1000], alpha = 0.5)
        plt.scatter(d1[-1000:], d2[-1000:], alpha = 0.5)
        plt.scatter(d1[1000:-1000], d2[1000:-1000], alpha = 0.5)
        plt.show()

        quit()


        return self._fit(x_train, x_test)

    def _fit(self, x, x_test = None):
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        batch_size = self._get_batch_size()
        n_batches = ceil(self.n_samples, batch_size)

        data = tf.placeholder(tf.float32, [None, self.n_features])

        # Create the dataset iterator
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size = self.n_samples)
        dataset = dataset.batch(batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        tf_data = iterator.get_next()

        make_encoder = tf.make_template('encoder', self._make_encoder)
        make_decoder = tf.make_template('decoder', self._make_decoder)

        # Define the model.
        prior = self._make_prior()
        posterior, post_means = make_encoder(data)
        code = posterior.sample()#self.n_samples)

        # Define the loss.
        # likelihood ~ E[log(P(X|z))]. This is approximated by a single sample
        # of z. In principle we could draw several samples to get a better approximation.
        likelihood = make_decoder(code).log_prob(data)
        # divergence ~ D_KL[Q(z|X) || P(z)] 
        divergence = tfd.kl_divergence(posterior, prior)
        # elbo ~ lower bound of log(P(X))
        elbo = tf.reduce_mean(likelihood / self.n_samples - divergence)
        optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(-elbo)

        #samples = make_decoder(prior.sample(10), [28, 28]).mean()

        # Initialize
        init = tf.global_variables_initializer()
        iterator_init = iterator.make_initializer(dataset)
        quit()

        with tf.train.MonitoredSession() as sess:
            sess.run(init)
            for it in range(self.n_iterations):
                test_cost = sess.run(elbo, feed_dict = {data:x_test})
                print("Test cost at iteration %d: %6.3f" % (it+1, test_cost))
                sess.run(iterator_init, feed_dict={data:x})
                for _ in range(n_batches):
                    sess.run(optimize)
            sess.run(iterator_init, feed_dict={data:x})
            z_means = []
            for _ in range(n_batches):
                z_means_batch = sess.run(post_means)
                z_means.append(z_means_batch)

        return z_means

