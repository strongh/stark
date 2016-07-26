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

        m = f1.shape[0] # number of parameters
        # I ignore the prior covariance
        sigma_j = [0]*m
        sigma_prior = np.identity(m)
        fs = [f1, f2]
        sigma_precision = np.linalg.inv(sigma_prior)
        prec_j = fs

        for j in [0, 1]:
            sigma_j[j] = np.cov(fs[j])
            prec_j[j] = np.linalg.inv(sigma_j[j] + sigma_precision)
        ## maybe could use matrix inversion lemma to do this correctly in pairwise fashion
        sigma = np.linalg.inv(sigma_precision + prec_j[0] + prec_j[1])

        w_j = [0]*m
        for j in [0, 1]:
            w_j[j] = np.dot(sigma, sigma_precision/J + prec_j[j])

        return np.dot(w_j[0], f1) + np.dot(w_j[1], f2)
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
        consensusSamples = subposteriors.reduce(consensus_avg(self.n_partitions))
        return consensusSamples

    def distribute(self, n=2):
        self.rdd = self.rdd.coalesce(1)
        single_rdd = self.rdd
        for i in range(n-1):
            self.rdd = self.rdd.union(single_rdd)
        posteriors = self.rdd.mapPartitions(mcmc(self.prepare_data_callback))

        return posteriors.reduce(concatenate_samples)
