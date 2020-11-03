# Infer

**The code in this repo does not work yet, it is at an API design stage.**

I am currently using this library as a way to test-drive mcx, fix bugs
and get a feel for what is currently missing. Use at your own risks, no
guarantee that anything will work for now!

# Goal

In its simplest form, `infer` can be simply a repository of models defined
as generative functions. It doesn't require much more work to pack the models in
a friendlier interface ML practitioners are used to. We can indeed wrap the
model definition, the sampler, the prior and posterior predictive samplers in a
scikit-learn like API.

`mcx`'s audience is data scientists who are curious about Bayesian methods
but not familiar enough to build custom models. We hope this will be both useful
and an incentive to dive into bayesian statistics more in depth.

# Tentative list of requirements 

An infer model is made of:
- A MCX model;
- A graphical representation (using daft);
- A mathematical representation;
- A default evaluator.

While it is useful to compute posterior distributions, it is critical to get a
point estimate for values in which we are interested. For that we can use the
Bayesian decision theory: implement a few loss functions and a pipeline that
goes from model to decision.

Model criticism is also very important in ML, so it is critical to provide tools
such as loo validation, [population predictive
checks](https://arxiv.org/abs/1908.00882) but also methods to compare models.
These tools should be usable for other mcx models than `infer` models.

Once we have all this we have a true Machine Learning library!

Murphy's *Machine Learning* is a good first repo of models that could be
implemented. Notebooks need to be written as ais it was pedagogical support for
a class on the topic. 

# Note for skeptics  

There's a lot to be said against this kind of library, but the same arguments
apply to traditional ML libraries and yet everyone is using them. Maybe this is
actually a way to get people interested in Bayesian methods and careful
statistical analysis?
