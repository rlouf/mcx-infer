from abc import ABC, abstractproperty, abstractmethod

import jax
import mcx


class Model(ABC):

    def __init__(self):
        self.rng_key = jax.random.PRNGKey(0)
        self.trace = None

    def __repr__(self):
        return self.math_repr

    def prior_predict(self, *args, num_samples=1000, **kwargs):
        """We should also be able to pass the data as simple args"""
        return mcx.predict(self, self.model)(**kwargs)

    def predict(self, *args, num_samples=1000, **kwargs):
        """We should also be able to pass the data as simple args"""
        if not self.trace:
            raise ValueError("""You must run the `.fit` method before being able to make predictions. Maybe you were looking for `prior_predict`?""")
        return mcx.predict(self, self.model, self.trace)(**kwargs)

    @abstractmethod
    def fit(self):
        pass

    def _fit(self, kernel, num_samples=1000, accelerate=True, **observations):
        """While it impossible to provide a universal fitting mechanism, some
        are certainly better than others.
        """

        _, self.rng_key = jax.random.split(self.rng_key)
        sampler = mcx.sampler(
                self.rng_key,
                self.model,
                kernel,
                **observations,
        )
        trace = sampler.run(1000, accelerate)

        self.sampler = sampler
        self.trace = trace

        return trace

    @abstractproperty
    def model(self):
        pass

    @abstractproperty
    def math_repr(self):
        pass

    @abstractproperty
    def graph(self):
        pass
