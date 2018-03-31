import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder


def projection_onto_simplex(b, f=1):
    feature = b.shape[0]
    a = np.sort(b)[::-1]
    csv = np.cumsum(a) - f
    indices = np.arange(feature) + 1
    condition = a - csv / indices > 0
    r = indices[condition][-1]
    teta = csv[condition][-1] / float(r)
    c = np.maximum(b - teta, 0)
    return c


class SVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, maximum_iterations=50, tol=0.05,
                 rndm_st=0, verb=0):
        self.C = C
        self.maximum_iterations = maximum_iterations
        self.tol = tol
        self.rndm_st = rndm_st
        self.verb = verb

    def _part_grad(self, D, e, x):
        # Partial gradient for the xth sample.
        i = np.dot(D[x], self.coef_.T) + 1
        i[e[x]] -= 1
        return i

    def _violations(self, i, e, x):
        # Optimality violation for the xth sample.
        small = np.inf
        for j in range(i.shape[0]):
            if j == e[x] and self.dual_coef_[j, x] >= self.C:
                continue
            elif j != e[x] and self.dual_coef_[j, x] >= 0:
                continue

            small = min(small, i[j])

        return i.max() - small

    def _solve_sub_problem(self, i, e, norm, x):
        # Prepare inputs to the projection.
        Cx = np.zeros(i.shape[0])
        Cx[e[x]] = self.C
        beta_hat = norm[x] * (Cx - self.dual_coef_[:, x]) + i / norm[x]
        f = self.C * norm[x]

        # Compute projection onto the simplex.
        beta = projection_onto_simplex(beta_hat, f)

        return Cx - self.dual_coef_[:, x] - beta / norm[x]

    def fits(self, D, e):
        n_samples, feature = D.shape

        # Normalize labels.
        self._label_encoder = LabelEncoder()
        e = self._label_encoder.fit_transform(e)

        # Initialize primal and dual coefficients.
        n_classes = len(self._label_encoder.classes_)
        self.dual_coef_ = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.coef_ = np.zeros((n_classes, feature))

        # Pre-compute norms.
        norm = np.sqrt(np.sum(D ** 2, axis=1))

        # Shuffle sample indices.
        rand_st = check_random_state(self.rndm_st)
        indices = np.arange(n_samples)
        rand_st.shuffle(indices)

        violation_initialization = None
        for iter in range(self.maximum_iterations):
            violation_sum = 0

            for xx in range(n_samples):
                x = indices[xx]

                # All-zero samples can be safely ignored.
                if norm[x] == 0:
                    continue

                i = self._part_grad(D, e, x)
                b = self._violations(i, e, x)
                violation_sum += b

                if b < 1e-12:
                    continue

                # Solve subproblem for the xth sample.
                delt = self._solve_sub_problem(i, e, norm, x)

                # Update primal and dual coefficients.
                self.coef_ += (delt * D[x][:, np.newaxis]).T
                self.dual_coef_[:, x] += delt

            if iter == 0:
                violation_initialization = violation_sum

            violationratio = violation_sum / violation_initialization

            if violationratio < self.tol:
                break

        return self

    def prediction(self, D):
        deci = np.dot(D, self.coef_.T)
        predict = deci.argmax(axis=1)
        return self._label_encoder.inverse_transform(predict)
