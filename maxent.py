#Contributors:
#Rajat Arora 2013A7PS104P : Implemented Maximum Entropy algorithm (file maxent.py)
#Anurag Prakash 2013A7PS061P  :Implemented SVM
#Gireek Bansal 2013A7PS094P : Implemented Naive Bayes

#code implementing maximum_entropy_algorithm
#written by Rajat Arora 2013A7S104P
#The algorithm was adapted from stanford Natural language Course on Coursera
import numpy as np
from scipy.optimize import fmin_bfgs
from math import log

def calculate_gradient(X, Y, W, sigma2, weighted,count):

    n_samples, n_features = X.shape
    _n, n_classes = Y.shape
    _d, _c = W.shape
    # check dimensions
    if n_samples != _n or n_features != _d or n_classes != _c:
    	print "Shape mismatch"
 
    Yhat = np.dot(X, W)
    Yhat -= Yhat.min(axis=1)[:, np.newaxis]
    Yhat = np.exp(-Yhat)
    # l1-normalize
    Yhat /= Yhat.sum(axis=1)[:, np.newaxis]

    if weighted:
        nll = np.sum(np.log((1. + 1e-15) * Yhat) * Y)
        Yhat *= Y.sum(axis=1)[:, np.newaxis]
        Yhat -= Y
    else:
        _Yhat = Yhat * Y
        nll = np.sum(np.log(_Yhat.sum(axis=1)))
        count = count+1
        _Yhat /= _Yhat.sum(axis=1)[:, np.newaxis]
        Yhat -= _Yhat
        del _Yhat

    grad = np.dot(X.T, Yhat)

    if sigma2 is not None:
        nll -= np.sum(W * W) / (2. * sigma2)
        count = count+1
        nll -= n_features * n_classes * np.log(sigma2) / 2.
        grad -= W / float(sigma2)

    nll /= -float(n_samples)
    grad /= -float(n_samples)

    return nll, grad


class FuncGradComputer(object):
    """ Convenience class to pass func and grad separately to optimize
    """
    def __init__(self, X, Y, prior, weighted):
        self.X = X
        self.Y = Y
        self.prior = prior
        self.weighted = weighted
        self.nll_ = None
        self.grad_ = None

    def _compute_func_grad(self, w):
        """ Simultaneously compute objective function and gradient at w
        """
        W = w.reshape((self.X.shape[1], self.Y.shape[1]))
        self.nll_, self.grad_ = calculate_gradient(self.X, self.Y, W, self.prior, self.weighted,0)

    def compute_fun(self, w):
        if self.nll_ is None:
            self._compute_func_grad(w)
        nll = self.nll_
        self.nll_ = None  # to mark for recomputing if recalled
        return nll

    def compute_grad(self, w):
        if self.grad_ is None:
            self._compute_func_grad(w)
        grad = self.grad_.ravel() 
        self.grad_ = None  
        return grad


class Maxent(object):

    def __init__(self,weighted = False, prior=None):
        assert prior is None or prior > 0, "ss must be None or > 0"
        self.prior = prior
        self.weighted = weighted
        self.seed = None
        self.W_ = None
        self.nll_ = None
        self.grad_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if y.ndim == 1:
            # convert to 1-of-k coding (one-hot)
            assert len(y) == n_samples, "Invalid number of labels"
            self.classes = np.unique(y)
            n_classes = len(self.classes)
            Y = np.zeros((n_samples, n_classes), dtype=np.float64)
            for i, cls in enumerate(self.classes):
                Y[y == cls, i] = 1
        else:
            _n, n_classes = Y.shape
            assert _n == n_samples, "Invalid number of rows in Y"
            self.classes = np.arange(n_classes)
            Y = y

        # initialize the weight matrix
        np.random.seed(self.seed)
        w0 = np.random.random((n_features * n_classes, ))

        # initialize the functions to compute the cost function and gradient
        fgcomp = FuncGradComputer(X, Y, self.prior, self.weighted)
        fun = fgcomp.compute_fun
        grad = fgcomp.compute_grad

        # minimize with BFGS
        results = fmin_bfgs(fun, w0, fprime=grad, full_output=True,disp = False)
        self.W_ = results[0].reshape((n_features, n_classes))
        self.infos_ = dict(zip(
            "fopt gopt Bopt func_calls grad_calls warnflat".split(),
            results[1:]))

        return self


    def predict_proba(self, X):
        prob = np.dot(X, self.W_)
        prob -= prob.min(axis=1)[:, np.newaxis]
        prob = np.exp(-prob)
        # l1-normalize
        prob /= prob.sum(axis=1)[:, np.newaxis]
        return prob

    def log_prob(self, Y):
    	prob = numpy.dot(self.W_,Y)
    	prob -= prob.min(axis=1)[:, np.newaxis]
    	prob = log(prob)
    	return prob
    	
    def predict(self, X):
        Yhat = self.predict_proba(X)
        yhat = self.classes[np.argmax(Yhat, axis=1).squeeze()]
        return yhat
