import numpy as np
import theano
import theano.tensor as tt
import theano.tensor.slinalg as tsl

A, C, G, T = range(4) # Encoding used in HKY paper 


def make_tensor(x):
	return tt.stacklists(x)

def hky_transition_probs(kappa, pi, t):
	piY = pi[T] + pi[C]
	piR = pi[A] + pi[G]

	beta = -1 / (2.0 * (piR*piY + kappa * (pi[A]*pi[G] + pi[C]*pi[T])))
	A_R = 1.0 + piR * (kappa - 1)
	A_Y = 1.0 + piY * (kappa - 1)
	lambd = make_tensor([ # Eigenvalues 
		0,
		beta,
		beta * A_Y,
		beta * A_R
	])
	U = make_tensor([ # Right eigenvectors as columns (rows of transpose)
		[1, 1, 1, 1],
		[1/piR, -1/piY, 1/piR, -1/piY],
		[0, pi[T]/piY, 0, -pi[C]/piY],
		[pi[G]/piR, 0, -pi[A]/piR, 0]
	]).transpose()

	Vt = make_tensor([ # Left eigenvectors as rows
		[pi[A], pi[C], pi[G], pi[T]],
		[pi[A]*piY, -pi[C]*piR, pi[G]*piY, -pi[T]*piR],
		[0, 1, 0, -1],
		[1, 0, -1, 0]
	])

	diag = tt.AllocDiag()(tt.exp(tt.outer(t, lambd)))
	U_dot_diag = tt.tensordot(U, diag, axes = 1)
	U_dot_diag_dot_Vt = tt.tensordot(U_dot_diag.transpose(), Vt, axes = [1, 0]).dimshuffle(1, 2, 0)

	return U_dot_diag_dot_Vt

def hky_transition_probs_scalar(kappa, pi, t):
	piY = pi[T] + pi[C]
	piR = pi[A] + pi[G]

	beta = -1 / (2.0 * (piR*piY + kappa * (pi[A]*pi[G] + pi[C]*pi[T])))
	A_R = 1.0 + piR * (kappa - 1)
	A_Y = 1.0 + piY * (kappa - 1)
	lambd = make_tensor([ # Eigenvalues 
		0,
		beta,
		beta * A_Y,
		beta * A_R
	])
	U = make_tensor([ # Right eigenvectors as columns (rows of transpose)
		[1, 1, 1, 1],
		[1/piR, -1/piY, 1/piR, -1/piY],
		[0, pi[T]/piY, 0, -pi[C]/piY],
		[pi[G]/piR, 0, -pi[A]/piR, 0]
	]).transpose()

	Vt = make_tensor([ # Left eigenvectors as rows
		[pi[A], pi[C], pi[G], pi[T]],
		[pi[A]*piY, -pi[C]*piR, pi[G]*piY, -pi[T]*piR],
		[0, 1, 0, -1],
		[1, 0, -1, 0]
	])

	return tt.dot(U, tt.dot(tt.diag(tt.exp(lambd * t)), Vt))


def hky_transition_probs_expm(kappa, pi, t):
	Q_nodiag = make_tensor([
		[0, kappa*pi[C], pi[A], pi[G]],
		[kappa*pi[T], 0, pi[A], pi[G]],
		[pi[T], pi[C], 0, kappa*pi[G]],
		[pi[T], pi[C], kappa*pi[A], 0]	
	])
	
	Q_unnormalised = Q_nodiag - tt.diag(Q_nodiag.sum(axis = 1))
	average_subs = -tt.nlinalg.trace(tt.dot(Q_unnormalised, tt.diag(pi)))

	return tsl.expm(Q * t / average_subs)

