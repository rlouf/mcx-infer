# MCX-learn

**The code in this repo does not work yet, it is at an API design stage.**

This is an experiment to see how a Bayesian Machine Learning library with MCX
would look like. It turns out that it would be very simple to build a library of
models with MCX. In its simplest form it could just be a collection of model
definitions, functions. However, if we want to have a "friendlier" interface
machine learning practicioners are used too we can simply wrap the model
definition, sampler, prior and posterior predictive samplers in a scikit-learn
like API.

We hope that data scientists who are curious about Bayesian methods but not
familiar enough to build custom models will find this useful.

We need, among other things:
- A reasonable default evaluator (NUTS) but the possibility to override that choice;
- The possibility to inject priors in the model. That'd be cool;
- return LateX expression of the model;
- display representation of graphical model using
  [daft](https://docs.daft-pgm.org/en/latest/);
- Provide point estimate using loss functions;

Murphy's *Machine Learning* is a good first repo of models that could be
implemented. Notebooks need to be written as ais it was pedagogical support for
a class on the topic. 

*Note for skeptics:* There's a lot to be said against this kind of library, but
the same arguments apply to traditional ML libraries and yet everyone is using
them. Maybe this is actually a way to get people interested in Bayesian methods
and careful statistical analysis?

