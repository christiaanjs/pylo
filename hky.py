import numpy as np
import theano
import theano.tensor as tt
import theano.tensor.slinalg as tsl

T, A, C, G = range(4)

def make_tensor(x):
	return tt.stacklists(x)

def hky_transition_probs(kappa, pi, t):
	piY = pi[T] + pi[C]
	piR = pi[A] + pi[G]
	lambd = make_tensor([ # Eigenvalues 
		0,
		-1,
		-(piY + piR*kappa),
		-(piY*kappa + piR)		
	])
	U = make_tensor([ # Right eigenvectors as columns (rows of transpose)
		[1, 1, 1, 1],
		[1/piY, 1/piY, -1/piR, -1/piR],
		[0, 0, pi[G]/piR, -pi[A]/piR],
		[pi[C]/piY, -pi[T]/piY, 0, 0]
	]).transpose()

	Vt = make_tensor([ # Left eigenvectors as rows
		[pi[T], pi[C], pi[A], pi[G]],
		[piR*pi[T], piR*pi[C], -piY*pi[A], -piY*pi[G]],
		[0, 0, 1, -1],
		[1, -1, 0, 0]
	])

	Q_unnormalised = tt.dot(U, tt.dot(tt.diag(lambd), Vt))
	average_subs = -tt.nlinalg.trace(tt.dot(Q_unnormalised, tt.diag(pi)))
	lambd_time_outer = tt.outer(t, lambd)
	diag = tt.AllocDiag()(tt.exp(lambd_time_outer / average_subs))
	U_dot_diag = tt.tensordot(U, diag, axes = 1)
	U_dot_diag_dot_Vt = tt.tensordot(U_dot_diag.transpose(), Vt, axes = [1, 0]).dimshuffle(1, 2, 0)

	return U_dot_diag_dot_Vt

def hky_transition_probs_scalar(kappa, pi, t):
	piY = pi[T] + pi[C]
	piR = pi[A] + pi[G]
	lambd = make_tensor([ # Eigenvalues 
		0,
		-1,
		-(piY + piR*kappa),
		-(piY*kappa + piR)		
	])
	U = make_tensor([ # Right eigenvectors as columns (rows of transpose)
		[1, 1, 1, 1],
		[1/piY, 1/piY, -1/piR, -1/piR],
		[0, 0, pi[G]/piR, -pi[A]/piR],
		[pi[C]/piY, -pi[T]/piY, 0, 0]
	]).transpose()

	Vt = make_tensor([ # Left eigenvectors as rows
		[pi[T], pi[C], pi[A], pi[G]],
		[piR*pi[T], piR*pi[C], -piY*pi[A], -piY*pi[G]],
		[0, 0, 1, -1],
		[1, -1, 0, 0]
	])

	Q_unnormalised = tt.dot(U, tt.dot(tt.diag(lambd), Vt))
	average_subs = -tt.nlinalg.trace(tt.dot(Q_unnormalised, tt.diag(pi)))

	return tt.dot(U, tt.dot(tt.diag(tt.exp(lambd / average_subs * t)), Vt))


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

