import jax.numpy as np

Model = object  # base class

# fmt: off
# flake8: noqa
class NaiveBayes(Model):
    """What's the point of all the fluff, instead of just returning the model?
    I'm note sure. Is it a *good thing* to simplify the procedure?
    """

    def __init__(self, num_categories):
        self.num_categories
        self.rng_key = jax.random.PRNGKey(0)

    def __repr_latex__(self):
        """LateX representation of the model for Jupyter notebooks."""
        pass

    def prior_predict(self, X):
        pass

    def predict(self, X, trace=None):
        if not self.trace:
            raise ValueError("""You must run the `.fit` method before being able to make predictions. Maybe you were looking for `prior_predict`?""")
        mcx.predict(X, self.trace)
        pass

    def fit(self, kernel, num_samples=1000, accelerate=True, **observations):
        """While it impossible to provide a universal fitting mechanism, some
        are certainly better than others.
        """
        model = self.model

        _, self.rng_key = jax.random.split(self.rng_key)
        sampler = mcx.sampler(
                self.rng_key,
                model,
                kernel,
                **observations,
        )
        trace = sampler.run(1000, accelerate)

        self.sampler = sampler
        self.trace = trace

        return trace


    def model(self):
        """Naive Bayes classifier.

        Note
        ----
        1. We shouldn't have to input the number of categories or number of predictors
        it should come from X's shape.
        2. Can we inject priors? That's all the fun
        3. prior_predict
        4. fit
        5. predict


        Actual Doc
        ----------

        We note :math:`x_{jc}` the value of the j-th element of the data vector :math:`x`
        conditioned on x belonging to the class :math:`c`. The Gaussian Naive Bayes
        algorithm models :math:`x_{jc}` as:

        .. math::
            x_{jc} \\sim Normal(\\mu_{jc}, \\sigma_{jc})

        While the probability that :math:`x` belongs to the class :math:`c` is given by the
        categorical distribution:

        .. math::
            P(y=c|x_i) = Cat(\\pi_1, \\dots, \\pi_C)

        where :math:`\\pi_i` is the probability that a vector belongs to category :math:`i`.
        We assume that the :math:`\\pi_i` follow a Dirichlet distribution:

        .. math::
            \\pi \\sim Dirichlet(\\alpha)

        with hyperparameter :math:`\\alpha = [1, .., 1]`. The :math:`\\mu_{jc}`
        are sampled from a Normal distribution centred on :math:`0` with
        variance :math:`100`, and the :math:`\\sigma_{jc}` are sampled from a
        HalfNormal distribuion of variance :math:`100`:

        .. math::
            \\mu_{jc} \\sim Normal(0, 100)
            \\sigma_{jc} \\sim HalfNormal(100)

        Note that the Gaussian Naive Bayes model is equivalent to a Gaussian
        mixture with a diagonal covariance [1].

        References
        ----------
        .. [1] Murphy, K. P. (2012). Machine learning: a probabilistic perspective.
        """
        
        @mcx.model
        def naive_bayes(X, num_categories):
            num_training_smaples, num_predictors = np.shape(X)

            # Priors
            alpha = np.ones(num_categories)
            pi <~ dist.Dirichlet(alpha, shape=num_categories)
            mu <~ dist.Normal(mu=0, sd=100, shape=(num_categories, num_predictors))
            sigma <~ dist.HalfNormal(100, shape=(num_categories, num_predictors))

            # Assign classes to data points
            z <~ dist.Categorical(pi, shape=num_training_samples)

            # The components are independent and normally distributed
            xi <~ dist.Normal(mu=mu[z], sd=sigma[z])

            return z

        return naive_bayes
