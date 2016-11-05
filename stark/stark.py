from pyspark import SparkContext, SparkConf, SparkFiles
import pystan
import numpy as np
from pystan import StanModel
import pickle

PICKLE_FILENAME = "stan_model.pkl"

def mcmc(callback):
    def w(sts):
        ## do MCMC...
        sts = list(sts)
        data = callback(sts)
        spst = SparkFiles.get(PICKLE_FILENAME)
        sm = pickle.load(open(spst, "rb"))
        fit = sm.sampling(data=data,
                          iter=2000, chains=4)
        h = [np.array(samples[1]) for samples in fit.extract().items()]
        for prm in h:
            if len(prm.shape)==1:
                prm.shape = (prm.shape[0], 1)
        # the singleton array is b/c we want to keep the matrix
        # together, not broken apart into rows
        return [np.hstack(h)]
    return w

def consensus_avg(J):
    def c(f1, f2):
        if np.isnan(f1).any():
            return f2

        # following Scott '16.
        # weights are optimal for Gaussian
        for j in [0, 1]:
            sigma_j[j] = np.cov(fs[j])

        return [
            sigma_j[0] + sigma_j[1],
            np.dot(sigma_j[0], f1) + np.dot(sigm_j[1], f2)
        ]
    return c


def concatenate_samples(a, b):
    return np.vstack((a, b))

class Stark:
    rdd = None
    n_partitions = None
    prepare_data_callback = None

    def __init__(self, context, rdd, prepare_data_callback, stan_file):
        self.rdd = rdd
        self.prepare_data_callback = prepare_data_callback
        self.n_partitions = self.rdd.getNumPartitions()

        sm = StanModel(file=stan_file)
        pickle.dump(sm, open(PICKLE_FILENAME, "wb"))
        context.addFile(PICKLE_FILENAME)


    def concensusWeight(self):
        subposteriors = self.rdd.mapPartitions(mcmc(self.prepare_data_callback))
        concensusProducts = subposteriors.reduce(consensus_avg(self.n_partitions))
        consensusSamples = np.dot(
            np.linalg.inv(concensusProducts[0]), # inverse of sum of the W_s
            concensusProducts[1] # sum of (W_s theta_s)
        )
        return consensusSamples

    def distribute(self, n=2):
        self.rdd = self.rdd.coalesce(1)
        single_rdd = self.rdd
        for i in range(n-1):
            self.rdd = self.rdd.union(single_rdd)
        posteriors = self.rdd.mapPartitions(mcmc(self.prepare_data_callback))

        return posteriors.reduce(concatenate_samples)
