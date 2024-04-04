"""
This script contains three tests.

The first test verifies that the bootstrapped means of the model applied to the parity violating data (i.e. the data
with signal) is equivalent to the bootstrapped means of the null dataset

The second test verifies that the bootstrapped means are equivalent to the cosmic variance, i.e. a full, independent
dataset with the same signal.

The third test verifies that everything is Gaussian, because this is an underlying assumption.

"""


from mocks import *

