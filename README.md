
Stark: Distributed Inference with Stan & Spark
===============

The idea of stark is to adapt a powerful single-node modeling tool ([Stan](http://mc-stan.org/)) for running on a distributed platform ([Spark](https://spark.apache.org/)). The naive way of doing this is to copy the data to each worker, and each worker produces poster samples. This emberassingly parallel model could be useful, but running in a distributed setting facilitates other possibilities, such as running on much larger datasets.

One strategy for handling more data is to run the same model where each one is provided a subset of data. Each model will then generate samples from a subposterior. Combining subposteriors is not trivial, and seems to be a current area of research. See the reading section for more resources.

Other possibilties could be running the same model with different datasets or hyperparameters (e.g. for a sensitivity analysis). If MAP estimates were found instead of full posteriors then one might have something akin to a distributed bootstrap.

**This project is experimental-quality**


Modes
========

The current status is that only the naive mode and simple weighted averaging are implemented.


Example
=======

Currently there is just a single example, built on the 8 schools dataset. Spark is not useful for such a small dataset, it is used just for illustration.

From the `stark/example` directory, try:

```
spark-submit stark_ex.py
```

(I've only tried this with Spark 1.6.2)

Reading
=======

+ [Patterns of Scalable Bayesian Inference](http://arxiv.org/abs/1602.05221)
+ [Bayes and Big Data:  The Consensus Monte Carlo Algorithm](http://research.google.com/pubs/pub41849.html)
+ [Asymptotically exact, embarrassingly parallel MCMC](https://arxiv.org/abs/1311.4780)


Related projects
=======

+ [parallelMCMCcombine](https://cran.r-project.org/web/packages/parallelMCMCcombine/index.html)
