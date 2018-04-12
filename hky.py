import numpy as np
import theano
import theano.tensor as tt
import theano.tensor.slinalg as tsl

T, A, C, G = range(4)

def make_tensor(x):
	return tt.stacklists(x)

def hky_transition_probs(alpha, beta, pi, t):
	piY = pi[T] + pi[C]
	piR = pi[A] + pi[G]
	lambd = make_tensor([ # Eigenvalues
		0,
		-beta,
		-(piY*beta + piR*alpha),
		-(piY*alpha + piR*beta)		
	])
	U = make_tensor([ # Right eigenvectors as columns
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

	return tt.dot(U, tt.dot(tt.diag(tt.exp(lambd * t)), Vt))


def hky_transition_probs_expm(alpha, beta, pi, t):
	Q_nodiag = make_tensor([
		[0, alpha*pi[C], beta*pi[A], beta*pi[G]],
		[alpha*pi[T], 0, beta*pi[A], beta*pi[G]],
		[beta*pi[T], beta*pi[C], 0, alpha*pi[G]],
		[beta*pi[T], beta*pi[C], alpha*pi[A], 0]	
	])

	Q = Q_nodiag - tt.diag(Q_nodiag.sum(axis = 1))
	return tsl.expm(Q*t)

# Start for one sequence
# Then vectorise (tensor)
def hky_loglike(alpha, beta, pi, t, x, y):
	"""
	alpha: Transition rate
	beta: Tranversion rate
	pi: Equilibrium frequencies
	x, y: Sequences (encoded as integers)
	t: Time
	"""
	Pt = hky_transition_probs(alpha, beta, pi, t)
	return Pt[x, y]
	
