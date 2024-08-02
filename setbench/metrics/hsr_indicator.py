import numpy as np
import math
from cvxopt import solvers, matrix
from copy import deepcopy

solvers.options['abstol'] = 1e-15
solvers.options['reltol'] = 1e-15
solvers.options['feastol'] = 1e-15
solvers.options['maxiters'] = 1000
solvers.options['show_progress'] = False


def sharpeRatio(p, Q, x, rf):
    """ Compute the Sharpe ratio.

    Returns the Sharpe ratio given the expected return vector, p,
    the covariance matrix, Q, the investment column vector, x, and
    the return of the riskless asset, rf.

    Parameters
    ----------
    p : ndarray
        Expected return vector (of size n).
    Q : ndarray
        Covariance (n,n)-matrix.
    x : ndarray
        Investment vector of size (n,1). The sum of which should be 1.
    rf : float
        Return of a riskless asset.

    Returns
    -------
    sr : float
        The HSR value.
    """
    return (x.T.dot(p) - rf) / math.sqrt(x.T.dot(Q).dot(x))


def _sharpeRatioQPMax(p, Q, rf):
    """ Sharpe ratio maximization problem - QP formulation """
    n = len(p)

    # inequality constraints (investment in assets is higher or equal to 0)
    C = np.diag(np.ones(n))
    d = np.zeros((n, 1), dtype=np.double)

    # equality constraints (just one)
    A = np.zeros((1, n), dtype=np.double)
    b = np.zeros((1, 1), dtype=np.double)
    A[0, :] = p - rf
    b[0, 0] = 1

    # convert numpy matrix to cvxopt matrix
    G, c, A, b, C, d = matrix(Q, tc='d'), matrix(np.zeros(n), tc='d'), matrix(A, tc='d'), matrix(b, tc='d'), matrix(C,
                                                                                                                    tc='d'), matrix(
        d, tc='d')

    sol = solvers.coneqp(G, c, -C, -d, None, A, b, kktsolver='ldl')  # , initvals=self.initGuess)
    y = np.array(sol['x'])

    return y


def sharpeRatioMax(p, Q, rf):
    """ Compute the Sharpe ratio and investment of an optimal portfolio.

    Parameters
    ----------
    p : ndarray
        Expected return vector (of size n).
    Q : ndarray
        Covariance (n,n)-matrix.
    rf : float
        Return of a riskless asset.

    Returns
    -------
    sr : float
        The HSR value.
    x : ndarray
        Investment vector of size (n,1).
    """
    y = _sharpeRatioQPMax(p, Q, rf)
    x = y / y.sum()
    x = np.where(x > 1e-9, x, 0)
    sr = sharpeRatio(p, Q, x, rf)
    return sr, x


# Assumes that l <= A << u
# Assumes A, l, u are numpy arrays
def _expectedReturn(A, l, u):
    """
    Returns the expected return (computed as defined by the HSR indicator), as a
    column vector.
    """
    A = np.array(A, dtype=np.double)  # because of division operator in python 2.7
    return ((u - A).prod(axis=-1)) / ((u - l).prod())


def _covariance(A, l, u, p=None):
    """  Returns the covariance matrix (computed as defined by the HSR indicator). """
    p = _expectedReturn(A, l, u) if p is None else p
    Pmax = np.maximum(A[:, np.newaxis, :], A[np.newaxis, ...])
    P = _expectedReturn(Pmax, l, u)

    Q = P - p[:, np.newaxis] * p[np.newaxis, :]
    return Q


def _argunique(pts):
    """ Find the unique points of a matrix. Returns their indexes. """
    ix = np.lexsort(pts.T)
    diff = (pts[ix][1:] != pts[ix][:-1]).any(axis=1)
    un = np.ones(len(pts), dtype=bool)
    un[ix[1:]] = diff
    return un


def HSRindicator(A, l, u, managedup=False):
    """
    Compute the HSR indicator of the point set A given reference points l and u.

    Returns the HSR value of A given l and u, and returns the optimal investment.
    By default, points in A are assumed to be unique.

    Tip: Either ensure that A does not contain duplicated points
        (for example, remove them previously and then split the
        investment between the copies as you wish), or set the flag
        'managedup' to True.

    Parameters
    ----------
    A : ndarray
        Input matrix (n,d) with n points and d dimensions.
    l : array_like
        Lower reference point.
    u : array_like
        Upper reference point.
    managedup : bool, optional
        If A contains duplicated points and 'managedup' is set to True, only the
        first copy may be assigned positive investment, all other copies are
        assigned zero investment. Otherwise, no special treatment is given to
        duplicate points.

    Returns
    -------
    hsri : float
        The HSR value.
       x : ndarray
        The optimal investment as a column vector array (n,1).
    """
    n = len(A)
    x = np.zeros((n, 1), dtype=float)

    # if u is not strongly dominated by l or A is the empty set
    if (u <= l).any():
        raise ValueError("The lower reference point does not strongly" \
                         "dominate the upper reference point!")

    if len(A) == 0:
        return 0, x

    valid = (A < u).all(axis=1)
    validix = np.where(valid)[0]

    # if A is the empty set
    if valid.sum() == 0:
        return 0, x
    A = A[valid]  # A only contains points that strongly dominate u
    A = np.maximum(A, l)
    m = len(A)  # new size (m <= n)

    # manage duplicate points
    ix = _argunique(A) if managedup else np.ones(m).astype(bool)
    p = _expectedReturn(A[ix], l, u)
    Q = _covariance(A[ix], l, u, p)

    hsri, x[validix[ix]] = sharpeRatioMax(p, Q, 0)

    return hsri, x


class HSR_Calculator:

    def __init__(self, lower_bound, upper_bound, max_obj_bool=None):
        '''
        Class to calculate HSR Indicator with assumption that assumes a maximization on all objectives.

         Parameters
        ----------
        lower_bound : array_like
            Lower reference point.
        upper_bound : array_like
            Upper reference point.
        max_obj_bool : bool, optional
            Details of the objectives for which dimension maximization is not the case.

        '''

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_obj_bool = None

        if max_obj_bool is not None:
            self.max_obj_bool = max_obj_bool


    def reset_lower_bound(self, lower_bound):
        self.lower_bound = lower_bound

    def reset_upper_bound(self, upper_bound):
        self.upper_bound = upper_bound

    def make_max_problem(self, matrix):

        if self.max_obj_bool is None:
            return matrix

        max_matrix = deepcopy(matrix)

        for dim in self.max_obj_bool:
            max_matrix[:, dim] = max_matrix**-1

        return max_matrix

    def calculate_hsr(self, solutions):

        max_solutions = self.make_max_problem(solutions)

        hsr_indicator, hsr_invest = HSRindicator(A=max_solutions, l=self.lower_bound, u=self.upper_bound)

        return hsr_indicator, hsr_invest


if __name__ == "__main__":

    # Example for 2 dimensions
    # Point set: {(1,3), (2,2), (3,1)},  l = (0,0), u = (4,4)
    A = np.array([[1, 3], [2, 2], [3, 1]])  # matrix with dimensions n x d (n points, d dimensions)
    l = np.zeros(2)  # l must weakly dominate every point in A
    u = np.array([4, 4])  # u must be strongly dominated by every point in A

    # A = np.array([[3.41e-01, 9.72e-01, 2.47e-01],
    #              [9.30e-01, 1.53e-01, 4.72e-01],
    #              [4.56e-01, 1.71e-01, 8.68e-01],
    #              [8.70e-02, 5.94e-01, 9.50e-01],
    #              [5.31e-01, 6.35e-01, 1.95e-01],
    #              [3.12e-01, 3.37e-01, 7.01e-01],
    #              [3.05e-02, 9.10e-01, 7.71e-01],
    #              [8.89e-01, 8.29e-01, 2.07e-02],
    #              [6.92e-01, 3.62e-01, 2.93e-01],
    #              [2.33e-01, 4.55e-01, 6.60e-01]])
    #
    # l = np.zeros(3)  # l must weakly dominate every point in A
    # u = np.array([1, 1, 1])


    hsr_class = HSR_Calculator(lower_bound=l, upper_bound=u)
    hsri, x = hsr_class.calculate_hsr(A)  # compute HSR indicator

    print("Optimal investment:")
    print("%s" % "\n".join(map(str, x[:, 0])))
    print("HSR indicator value: %f" % hsri)