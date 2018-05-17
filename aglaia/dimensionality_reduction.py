
# https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
# https://github.com/wiseodd/generative-models/blob/master/VAE/vanilla_vae/vae_tensorflow.py
# https://danijar.com/building-variational-auto-encoders-in-tensorflow/

import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

from .utils import InputError, is_positive_integer, ceil

class VAE(object):
    """
    Dimensionality reduction with variational autoencoders.
    The code is slightly modified from 
    https://danijar.com/building-variational-auto-encoders-in-tensorflow/

    """

    tfd = tf.contrib.distributions

    def __init__(self, dimensions = 2, layer_sizes = [200,200], learning_rate = 0.001,
            activation_function = tf.nn.relu, n_samples = 1, n_iterations = 20, batch_size = 100):
        self.dimensions = dimensions
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.batch_size = batch_size

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


    def _make_decoder(self, x):
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
            x = tf.layers.dense(x, layer_size, self.activation_function)

        #x = tf.layers.dense(x, self.n_features, tf.nn.relu)
        alpha = tf.layers.dense(x, self.dimensions, tf.nn.softplus)
        beta = tf.layers.dense(x, self.dimensions, tf.nn.softplus)
        return tfd.Independent(tfd.GammaDistribution(alpha, beta), 1)

    def fit(self, filenames):

        # TODO create slatm for _fit function

        return self._fit(x, x_test)

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
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output.shapes)
        tf_data = iterator.get_next()

        make_encoder = tf.make_template('encoder', self._make_encoder)
        make_decoder = tf.make_template('decoder', self._make_decoder)

        # Define the model.
        prior = self._make_prior()
        posterior, post_means = make_encoder(data)
        code = posterior.sample(self.n_samples)

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

    def __init__(self, representation = 'unsorted_coulomb_matrix', 
            slatm_sigma1 = 0.05, slatm_sigma2 = 0.05, slatm_dgrid1 = 0.03, slatm_dgrid2 = 0.03, slatm_rcut = 4.8, slatm_rpower = 6,
            slatm_alchemy = False, compounds = None, properties = None, **kwargs):
        """
        A molecule's cartesian coordinates and chemical composition is transformed into a descriptor for the molecule,
        which is then used as input to a single or multi layered feedforward neural network with a single output.
        This class inherits from the _NN and _ONN class and all inputs not unique to the OMNN class is passed to the
        parents.

        Available representations at the moment are ['unsorted_coulomb_matrix', 'sorted_coulomb_matrix',
        bag_of_bonds', 'slatm'].

        :param representation: Name of molecular representation.
        :type representation: string
        :param slatm_sigma1: Scale of the gaussian bins for the two-body term
        :type slatm_sigma1: float
        :param slatm_sigma2: Scale of the gaussian bins for the three-body term
        :type slatm_sigma2: float
        :param slatm_dgrid1: Spacing between the gaussian bins for the two-body term
        :type slatm_dgrid1: float
        :param slatm_dgrid2: Spacing between the gaussian bins for the three-body term
        :type slatm_dgrid2: float
        :param slatm_rcut: Cutoff radius
        :type slatm_rcut: float
        :param slatm_rpower: exponent of the binning
        :type slatm_rpower: integer
        :param slatm_alchemy: Whether to use the alchemy version of slatm or not.
        :type slatm_alchemy: bool

        """

        # TODO try to avoid directly passing compounds and properties. That shouldn't be needed.
        super(OMNN,self).__init__(compounds = compounds, properties = properties, **kwargs)

        self._set_representation(representation, slatm_sigma1, slatm_sigma2, slatm_dgrid1, slatm_dgrid2, slatm_rcut,
                slatm_rpower, slatm_alchemy)

    def _set_properties(self, properties):
        """
        Set properties. Needed to be called before fitting.

        :param y: array of properties of size (nsamples,)
        :type y: array
        """
        if not is_none(properties):
            if is_numeric_array(properties) and np.asarray(properties).ndim == 1:
                self.properties = np.asarray(properties)
            else:
                raise InputError('Variable "properties" expected to be array like of dimension 1. Got %s' % str(properties))
        else:
            self.properties = None

    def _set_representation(self, representation, *args):

        if not is_string(representation):
            raise InputError("Expected string for variable 'representation'. Got %s" % str(representation))
        if representation.lower() not in ['sorted_coulomb_matrix', 'unsorted_coulomb_matrix', 'bag_of_bonds', 'slatm']:
            raise InputError("Unknown representation %s" % representation)
        self.representation = representation.lower()

        self._set_slatm(*args)

    def _set_slatm(self, slatm_sigma1, slatm_sigma2, slatm_dgrid1, slatm_dgrid2, slatm_rcut,
            slatm_rpower, slatm_alchemy):

        if not is_positive(slatm_sigma1):
            raise InputError("Expected positive float for variable 'slatm_sigma1'. Got %s." % str(slatm_sigma1))
        self.slatm_sigma1 = float(slatm_sigma1)

        if not is_positive(slatm_sigma2):
            raise InputError("Expected positive float for variable 'slatm_sigma2'. Got %s." % str(slatm_sigma2))
        self.slatm_sigma2 = float(slatm_sigma2)

        if not is_positive(slatm_dgrid1):
            raise InputError("Expected positive float for variable 'slatm_dgrid1'. Got %s." % str(slatm_dgrid1))
        self.slatm_dgrid1 = float(slatm_dgrid1)

        if not is_positive(slatm_dgrid2):
            raise InputError("Expected positive float for variable 'slatm_dgrid2'. Got %s." % str(slatm_dgrid2))
        self.slatm_dgrid2 = float(slatm_dgrid2)

        if not is_positive(slatm_rcut):
            raise InputError("Expected positive float for variable 'slatm_rcut'. Got %s." % str(slatm_rcut))
        self.slatm_rcut = float(slatm_rcut)

        if not is_non_zero_integer(slatm_rpower):
            raise InputError("Expected non-zero integer for variable 'slatm_rpower'. Got %s." % str(slatm_rpower))
        self.slatm_rpower = int(slatm_rpower)

        if not is_bool(slatm_alchemy):
            raise InputError("Expected boolean value for variable 'slatm_alchemy'. Got %s." % str(slatm_alchemy))
        self.slatm_alchemy = bool(slatm_alchemy)

    def get_descriptors_from_indices(self, indices):

        if is_none(self.properties):
            raise InputError("Properties needs to be set in advance")
        if is_none(self.compounds):
            raise InputError("QML compounds needs to be created in advance")

        if not is_positive_integer_or_zero_array(indices):
            raise InputError("Expected input to be indices")

        # Convert to 1d
        idx = np.asarray(indices, dtype=int).ravel()

        if self.representation == 'unsorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))//2
            x = np.empty((idx.size, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_coulomb_matrix(size = nmax, sorting = "unsorted")
                x[i] = mol.representation

        if self.representation == 'sorted_coulomb_matrix':

            nmax = self._get_msize()
            representation_size = (nmax*(nmax+1))//2
            x = np.empty((idx.size, representation_size), dtype=float)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_coulomb_matrix(size = nmax, sorting = "row-norm")
                x[i] = mol.representation

        elif self.representation == "bag_of_bonds":
            asize = self._get_asize()
            x = np.empty(idx.size, dtype=object)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_bob(asize = asize)
                x[i] = mol.representation
            x = np.asarray(list(x), dtype=float)

        elif self.representation == "slatm":
            mbtypes = self._get_slatm_mbtypes([mol.nuclear_charges for mol in self.compounds])
            x = np.empty(idx.size, dtype=object)
            for i, mol in enumerate(self.compounds[idx]):
                mol.generate_slatm(mbtypes, local = False, sigmas = [self.slatm_sigma1, self.slatm_sigma2],
                        dgrids = [self.slatm_dgrid1, self.slatm_dgrid2], rcut = self.slatm_rcut, alchemy = self.slatm_alchemy,
                        rpower = self.slatm_rpower)
                x[i] = mol.representation
            x = np.asarray(list(x), dtype=float)

        return x

    # TODO test
    def fit(self, indices, y = None):
        """
        Fit the neural network to a set of molecular descriptors and targets. It is assumed that QML compounds and
        properties have been set in advance and which indices to use is given.

        :param y: Dummy for osprey
        :type y: None
        :param indices: Which indices of the pregenerated QML compounds and properties to use.
        :type indices: integer array

        """

        x = self.get_descriptors_from_indices(indices)

        idx = np.asarray(indices, dtype = int).ravel()
        y = self.properties[idx]

        return self._fit(x, y)

    def predict(self, indices):
        x = self.get_descriptors_from_indices(indices)
        return self._predict(x)
