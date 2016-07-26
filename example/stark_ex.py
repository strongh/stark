from pyspark import SparkContext, SparkConf
from stark import *

SCHOOL_DATA = zip(
    [28,  8, -3,  7, -1,  1, 18, 12], # y
    [15, 10, 16, 11,  9, 11, 10, 18]) # sigma

def prepare_school_data(data):
    return {'J': len(data),
            'y': [d[0] for d in data],
            'sigma': [d[1] for d in data]}


if __name__ == "__main__":
    stanFile = "schools.stan"
    conf = SparkConf().setAppName("ExampleStarkApp")
    sc = SparkContext(conf=conf)
    school_rdd = sc.parallelize(SCHOOL_DATA, 2)
    st = stark.Stark(sc, school_rdd, prepare_school_data, stanFile)
    ## 2 ways to distribute:
    ## + combine each subposterior, weighting according to (co)variances:
    weighted_avg = st.concensusWeight()
    ## + naive parallel, like running n*chains:
    dist = st.distribute(n=4)
    print "Weighted average posterior samples:"
    print weighted_avg
    print "Posteriors drawn from parallel workers:"
    print dist
