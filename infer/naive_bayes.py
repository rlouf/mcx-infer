import daft
import jax
import jax.numpy as np
import mcx
import mcx.distributions as dist
from matplotlib import rc
from IPython.display import display, Math

from infer.model import Model


class NaiveBayes(Model):
    """Naive Bayes classifier.

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

    Note
    ----
    MCX is in an integration testing phase and this example will likely not
    work as compilation may output garbage. Even if it did work we would need
    to be able to deal with the fact that `z` is a discrete random variable
    thus not sample-able by an evaluator in the HMC family.


    References
    ----------
    .. [1] Murphy, K. P. (2012). Machine learning: a probabilistic perspective.
    """

    def __init__(self, num_categories):
        super.__init__(self)
        self.num_categories = num_categories

    @property
    def model(self):

        # flake8: noqa
        @mcx.model
        def naive_bayes(X, num_categories):
            num_predictors = np.shape(X)[1]
            num_training_samples = np.shape(X)[0]

            # Priors
            alpha = np.ones(num_categories)
            pi <~ dist.Dirichlet(alpha, shape=num_categories)
            mu <~ dist.Normal(mu=0, sigma=100, shape=(num_categories, num_predictors))
            sigma <~ dist.Exponential(100, shape=(num_categories, num_predictors))

            # Assign classes to data points
            z <~ dist.Categorical(pi, shape=num_training_samples)

            # The components are independent and normally distributed
            xi <~ dist.Normal(mu=mu[z], sd=sigma[z])

            return z

        return naive_bayes

    def fit(self, kernel=None, num_samples=1000, accelerate=False, **observations):
        """Fit the Naive Bayes model.

        The kernel is currently set to a HMC sampler for all variables; however 'z' is
        a discrete variable and this will need to be changed in the future.

        This paradigm allows to specify a default (explicit) evaluator while letting
        the user experiment with others.

        """
        kwargs = dict({'num_categories': num_categories}, **observations)
        if not kernel:
            kernel = HMC(30)
        trace = self._fit(kernel, num_samples, accelerate, **kwargs)

        return trace
    
    @property
    def math_repr(self):
        """LateX representation of the model for Jupyter notebooks."""
        representation = r"""
        \begin{align}
        \mu_{jc} & \sim \text{Normal}(0, 100) \\
        \sigma_{jc} &\sim \text{HalfNormal}(100)\\
        x_{jc} & \sim \text{Normal}(\mu_{jc}, \sigma_{jc}) \\
        \pi & \sim \text{Dirichlet}(\alpha)\\
        P(y=c|x_i) &= \text{Cat}(\pi_1, \dots, \pi_C)
        \end{align}
        """
        return display(Math(representation))
    
    @property
    def graph(self):
        """This is just an example, not the actual model.
        """
        rc("font", family="serif", size=12)
        rc("text", usetex=True)

        # Colors.
        p_color = {"ec": "#46a546"}
        s_color = {"ec": "#f89406"}

        pgm = daft.PGM()

        n = daft.Node("phi", r"$\phi$", 1, 3, plot_params=s_color)
        n.va = "baseline"
        pgm.add_node(n)
        pgm.add_node("speckle_coeff", r"$z_i$", 2, 3, plot_params=s_color)
        pgm.add_node("speckle_img", r"$x_i$", 2, 2, plot_params=s_color)

        pgm.add_node("spec", r"$s$", 4, 3, plot_params=p_color)
        pgm.add_node("shape", r"$g$", 4, 2, plot_params=p_color)
        pgm.add_node("planet_pos", r"$\mu_i$", 3, 3, plot_params=p_color)
        pgm.add_node("planet_img", r"$p_i$", 3, 2, plot_params=p_color)

        pgm.add_node("pixels", r"$y_i ^j$", 2.5, 1, observed=True)

        # Edges.
        pgm.add_edge("phi", "speckle_coeff")
        pgm.add_edge("speckle_coeff", "speckle_img")
        pgm.add_edge("speckle_img", "pixels")

        pgm.add_edge("spec", "planet_img")
        pgm.add_edge("shape", "planet_img")
        pgm.add_edge("planet_pos", "planet_img")
        pgm.add_edge("planet_img", "pixels")

        # And a plate.
        pgm.add_plate([1.5, 0.2, 2, 3.2], label=r"exposure $i$", shift=-0.1)
        pgm.add_plate([2, 0.5, 1, 1], label=r"pixel $j$", shift=-0.1)

        # Render and save.
        pgm.render(dpi=120)
