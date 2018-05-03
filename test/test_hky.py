from pylo.hky import *
import theano
import theano.tensor as tt
import numpy as np
from numpy.testing import assert_allclose
import pytest

test_data = [
	(
		2.0,
		np.array([0.25, 0.25, 0.25, 0.25]),
		0.1,
		np.array([0.906563342722, 0.023790645491, 0.045855366296, 0.023790645491,
                    0.023790645491, 0.906563342722, 0.023790645491, 0.045855366296,
                    0.045855366296, 0.023790645491, 0.906563342722, 0.023790645491,
                    0.023790645491, 0.045855366296, 0.023790645491, 0.906563342722]).reshape(4, 4)
	),(
		2.0,
		np.array([0.50, 0.20, 0.2, 0.1]),
		0.1,
		np.array([0.928287993055, 0.021032136637, 0.040163801989, 0.010516068319,
                    0.052580341593, 0.906092679369, 0.021032136637, 0.020294842401,
                    0.100409504972, 0.021032136637, 0.868042290072, 0.010516068319,
                    0.052580341593, 0.040589684802, 0.021032136637, 0.885797836968]).reshape(4, 4)
	),(
		5.0,
		np.array([0.20, 0.30, 0.25, 0.25]),
		0.1,
		np.array([0.904026219693, 0.016708646875, 0.065341261036, 0.013923872396,
                    0.011139097917, 0.910170587813, 0.013923872396, 0.064766441875,
                    0.052273008829, 0.016708646875, 0.917094471901, 0.013923872396,
                    0.011139097917, 0.077719730250, 0.013923872396, 0.897217299437]).reshape(4, 4)
	)
]

@pytest.mark.parametrize('kappa,pi,t,expected', test_data)
def test_hky_transition_probs_scalar(kappa, pi, t, expected):
	kappa_ = tt.scalar()
	pi_ = tt.vector()
	t_ = tt.scalar()
	
	transition_probs_ = hky_transition_probs_scalar(kappa_, pi_, t_)
	
	f = theano.function([kappa_, pi_, t_], transition_probs_)
	res = f(kappa, pi, t)
	assert_allclose(res, expected)


def test_hky_transition_probs_vec():
	kappa_ = tt.scalar()
	pi_ = tt.vector()
	t_ = tt.scalar()
	ts_ = tt.vector()

	kappa = 1.2
	pi = np.array([0.3, 0.2, 0.25, 0.25])
	ts = np.array([1.2, 0.8, 1.3])
	
	transition_probs_ = hky_transition_probs_scalar(kappa_, pi_, t_)
	transition_probs_vectorised_ = hky_transition_probs_vec(kappa_, pi_, ts_)	

	f = theano.function([kappa_, pi_, t_], transition_probs_)
	f_vectorised = theano.function([kappa_, pi_, ts_], transition_probs_vectorised_)
	res_scalar = [f(kappa, pi, t) for t in ts]
	res_vectorised = f_vectorised(kappa, pi, ts)
	assert_allclose(res_vectorised, np.stack(res_scalar, axis=0))

def test_hky_transition_probs_mat():
	kappa_ = tt.scalar()
	pi_ = tt.vector()
	t_ = tt.scalar()
	ts_ = tt.matrix()

	kappa = 1.2
	pi = np.array([0.3, 0.2, 0.25, 0.25])
	ts = np.array([[1.2, 0.8, 1.3], [1.1, 0.7, 2.2]])
	
	transition_probs_ = hky_transition_probs_scalar(kappa_, pi_, t_)
	transition_probs_vectorised_ = hky_transition_probs_mat(kappa_, pi_, ts_)	

	f = theano.function([kappa_, pi_, t_], transition_probs_)
	f_vectorised = theano.function([kappa_, pi_, ts_], transition_probs_vectorised_)
	res_scalar = [[f(kappa, pi, t) for t in trow] for trow in ts]
	res_vectorised = f_vectorised(kappa, pi, ts)
	assert_allclose(res_vectorised, np.stack(res_scalar, axis=0))
