# Infer

**The code in this repo does not work yet, it is at an API design stage.**

I am currently using this library as a way to test-drive the library, fix bugs
and get a feel for what is currently missing. Use at your own risks, no
guarantee that anything will work for now!

# Goal

In its simplest form, `mcx-infer` can be simply a repository of models defined
as generative functions. It doesn't require much more work to pack the models in
a friendlier interface ML practitioners are used to. We can indeed wrap the
model definition, the sampler, the prior and posterior predictive samplers in a
scikit-learn like API.

`mcx-infer`'s audience is data scientists who are curious about Bayesian methods
but not familiar enough to build custom models. We hope this will be both useful
and an incentive to dive into bayesian statistics more in depth.

# Tentative list of requirements 

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

# Note for skeptics  

There's a lot to be said against this kind of library, but the same arguments
apply to traditional ML libraries and yet everyone is using them. Maybe this is
actually a way to get people interested in Bayesian methods and careful
statistical analysis?
