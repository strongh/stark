from pyspark import SparkContext, SparkConf
from stark import *
import pystan
import numpy as np
import unittest

SCHOOL_DATA = zip(
    [28,  8, -3,  7, -1,  1, 18, 12], # y
    [15, 10, 16, 11,  9, 11, 10, 18]) # sigma

def prepare_school_data(data):
    return {'J': len(data),
            'y': [d[0] for d in data],
            'sigma': [d[1] for d in data]}


schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}


stanFile = "schools.stan"
## first run stan in the standard way

fit = pystan.stan(file="schools.stan", data=schools_dat, iter=2000, chains=4)
param_means = np.mean(fit.get_posterior_mean(), 1) # these are the "targets"
## now try the spark version
conf = SparkConf().setAppName("ExampleStarkApp")
sc = SparkContext(conf=conf)
school_rdd = sc.parallelize(SCHOOL_DATA, 2)
st = stark.Stark(sc, school_rdd, prepare_school_data, stanFile)

N_DISTRIBUTE = 4
sp_fit = st.distribute(n=N_DISTRIBUTE)
sp_param_means = np.mean(sp_fit, 0)


class TestDistribute(unittest.TestCase):
    def test(self): # close enough
        self.assertEqual((1000*N_DISTRIBUTE, 19), sp_fit.shape)
        self.assertTrue(np.linalg.norm(param_means - sp_param_means) < 1.5)


sp_wa = st.concensusWeight()
sp_wa_param_means = np.mean(sp_fit, 0)

class TestWeightedAvg(unittest.TestCase):
    def test(self): # close enough
        self.assertEqual((1000, 19), sp_wa.shape)
        self.assertTrue(np.linalg.norm(param_means - sp_wa_param_means) < 1.5)

if __name__ == '__main__':
    unittest.main()
