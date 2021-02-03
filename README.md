# Infer

**The code in this repo is a design draft, a proof of concept if you wish.**

I am currently using this library as a way to test-drive mcx, fix bugs
and get a feel for what is currently missing. Nevertheless, I really care about
this project and will make it a priority in the near future.

# Goal

Pack generative models in
a familiar, scikit-learn like, interface for Machine Learning practitioners. Wraps
the model, prior and posterior predictive samplers and posterior sampler.

`infer`'s audience is data scientists who are curious about Bayesian methods
but not familiar enough to build custom models, let alone choose the best inference
method. We hope this will be both useful and an incentive to dive into bayesian statistics
more in depth.

# Tentative list of requirements 

## Models

An infer model is made of:
- A MCX model;
- A graphical representation (using [daft](https://github.com/daft-dev/daft);
- A mathematical representation;
- A default evaluator. This evaluator can be tailored to the model being sampled
  so the user doesn't need to tinker with inference.

## Point estimates - Bayesian decision theory

While it is useful to compute posterior distributions, it is critical to get a
point estimate for values in which we are interested. For that we can use the
Bayesian decision theory: **implement a few loss functions and a pipeline that
goes from model to decision**.

## Model criticism

Model criticism is as important in ML as it is in statistics, so it is critical to provide tools
such as loo validation, [population predictive
checks](https://arxiv.org/abs/1908.00882) but also methods to compare models.
These tools should be usable for other MCX models than `infer` models.

Once we have all this we have a true Machine Learning library!

# What models?

Murphy's *Machine Learning* is a good first repo of models that could be
implemented. Notebooks need to be written *for each model* as if it were a pedagogical support for
a class on the topic. 

# Note for skeptics  

There's a lot to be said against this kind of library, but the same arguments
apply to traditional ML libraries and yet everyone is using them. Maybe this is
actually a way to get people interested in Bayesian methods and careful
statistical analysis?
