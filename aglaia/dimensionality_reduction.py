
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

from .utils import InputError, is_positive_integer

class VAE(object):
    """
    Dimensionality reduction with variational autoencoders.
    The code is slightly modified from 
    https://danijar.com/building-variational-auto-encoders-in-tensorflow/

    """

    tfd = tf.contrib.distributions

    def __init__(self, dimensions = 2, layer_sizes = [200,200]):
        self.dimensions = dimensions
        self.layer_sizes = layer_sizes

    def _make_encoder(self, data):
        """
        ReLU activation on hidden layers is used and softplus
        to force scale to be positive.
        """
        x = tf.layers.flatten(data)
        for layer_size in self.layer_sizes:
            x = tf.layers.dense(x, layer_size, tf.nn.relu)

        loc = tf.layers.dense(x, self.dimensions)
        scale = tf.layers.dense(x, self.dimensions, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)


    def _make_prior(self):
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
        A Deterministic distribution is used to get the entropy,
        but a full- or half-normal or something similar might be better.

        """

        for layer_size in self.layer_sizes:
            x = tf.layers.dense(x, layer_size, tf.nn.relu)

        x = tf.layers.dense(x, self.n_features, tf.nn.relu)
        x = tf.reshape(x, [-1, self.n_features])
        return tfd.Independent(tfd.Deterministic(x), 1)

    def fit(self, filenames):

        # TODO create slatm for _fit function

    def _fit(self, x):

        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]

        data = tf.placeholder(tf.float32, [None, self.n_features])

        make_encoder = tf.make_template('encoder', self._make_encoder)
        make_decoder = tf.make_template('decoder', self._make_decoder)

        # Define the model.
        prior = self._make_prior()
        posterior = make_encoder(data)
        code = posterior.sample()

        # Define the loss.
        likelihood = make_decoder(code).log_prob(data)
        divergence = tfd.kl_divergence(posterior, prior)
        elbo = tf.reduce_mean(likelihood - divergence)
        optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)

        samples = make_decoder(prior.sample(10), [28, 28]).mean()

        mnist = input_data.read_data_sets('MNIST_data/')
        fig, ax = plt.subplots(nrows=20, ncols=11, figsize=(10, 20))
        with tf.train.MonitoredSession() as sess:
          for epoch in range(20):
            feed = {data: mnist.test.images.reshape([-1, 28, 28])}
            test_elbo, test_codes, test_samples = sess.run([elbo, code, samples], feed)
            plot_codes(ax[epoch, 0], test_codes, mnist.test.labels)
            plot_samples(ax[epoch, 1:], test_samples)
            for _ in range(600):
              feed = {data: mnist.train.next_batch(100)[0].reshape([-1, 28, 28])}
              sess.run(optimize, feed)

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
