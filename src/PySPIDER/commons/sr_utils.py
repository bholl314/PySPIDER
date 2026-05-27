import numpy as np


def keep_inds(vector, inds):  # set all but inds of vector to 0
    inds = list(inds)
    temp = vector * 0
    temp[inds] = vector[inds]
    return temp


def smallest_sv(A, inds=None, value=False):
    # run with eps = 0 first, if that doesn't work, we use Tikhonov-regularization on increasing offsets
    for eps in (0, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            B = A
            if inds != None:
                B = A[:, inds]
            # stabilize A by adding eps*I to A^T A
            ATA = B.T @ B + eps * np.eye(B.shape[1])
            if not np.all(np.isfinite(B)):
                raise ValueError(
                    f"Non-finite values in B: "
                    f"{np.sum(np.isnan(B))} NaNs, "
                    f"{np.sum(np.isinf(B))} Infs, "
                    f"shape={B.shape}"
                    )
            U, Sigma, Vt = np.linalg.svd(ATA, full_matrices=True)
            if value:
                # Singular values of A are sqrt of eigenvalues of A^T A
                return np.sqrt(max(Sigma[-1], 0))
            else:
                V = Vt.T
                return V[:, -1]
        except np.linalg.LinAlgError:
            print("Failed for: " + str(eps))
            continue
    raise RuntimeError("smallest_sv failed for all Tikhonov regularization offsets")


    


def smallest_eig(A, inds=None, value=False):
    if inds is None:
        lambdas, vs = np.linalg.eig(A)
    else:
        lambdas, vs = np.linalg.eig(A[np.ix_(inds, inds)])
    # idx = lambdas.argsort()[::-1]
    # vs = vs[:,idx]
    idx = np.argmin(lambdas)
    if value:
        return lambdas[idx]
    else:
        return vs[:, idx]


# inhomogeneous regression is UNTESTED
# def solve_ATA(A, inds, inhomog_col=None):
#    w = A.shape[1]
#    x = np.zeros(shape=(w,))
#    if inhomog_col is None:
#        x[np.ix_(inds)] = smallest_sv(A[np.ix_(inds, inds)]) # work on submatrix with inds
#    else: # note that [A b]^T[A b] = [A^TA A^Tb; ... ...]
#        inds_minus_b = inds.copy()
#        inds_minus_b.remove(inhomog_col)
#        ATA = A[np.ix_(inds_minus_b, inds_minus_b)]
#        ATb = A[np.ix_(inds_minus_b, [inhomog_col])]
#        #x, _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)
#        x[np.ix_(inds_minus_b)] = np.linalg.solve(ATA, ATb)
#        x[inhomog_col] = -1 # put back in the -1 coefficient for b
#    return x
#
# def solve(A, inds, inhomog_col=None):
#    h, w = A.shape
#    x = np.zeros(shape=(w,))
#    if inhomog_col is None:
#        x[np.ix_(inds)] = smallest_sv(A[np.ix_(range(h), inds)]) # work on submatrix with inds
#    else: # note that [A b]^T[A b] = [A^TA A^Tb; ... ...]
#        inds_minus_b = inds.copy()
#        inds_minus_b.remove(inhomog_col)
#        ATA = A[np.ix_(range(h), inds_minus_b)]
#        ATb = A[np.ix_(range(h), [inhomog_col])]
#        x[np.ix_(inds_minus_b)], _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)
#        x[inhomog_col] = -1 # put back in the -1 coefficient for b
#    return x


def solve_ATA(A, inds, inhomog_col=None):  # A here is A^TA
    w = A.shape[1]
    x = np.zeros(shape=(w,))
    inds = list(inds)
    if inhomog_col is None:
        x[inds] = smallest_eig(A[np.ix_(inds, inds)])  # work on submatrix with inds
    else:  # note that [A b]^T[A b] = [A^TA A^Tb; ... ...]
        inds_minus_b = inds.copy()
        inds_minus_b.remove(inhomog_col)
        ATA = A[np.ix_(inds_minus_b, inds_minus_b)]
        ATb = A[np.ix_(inds_minus_b, [inhomog_col])]
        # x, _, _, _ = np.linalg.lstsq(ATA, ATb, rcond=None)
        x[inds_minus_b] = np.linalg.solve(ATA, ATb)[:, 0]
        x[inhomog_col] = -1  # put back in the -1 coefficient for b
    return x


def solve(A, inds, inhomog_col=None):
    w = A.shape[1]
    x = np.zeros(shape=(w,))
    inds = list(sorted(inds))
    if inhomog_col is None:
        x[inds] = smallest_sv(A[:, inds])  # work on submatrix with inds
    else:  # note that [A b]^T[A b] = [A^TA A^Tb; ... ...]
        inds_minus_b = inds.copy()
        inds_minus_b.remove(inhomog_col)
        A_submx = A[:, inds_minus_b]
        b = A[:, inhomog_col]
        x[inds_minus_b], _, _, _ = np.linalg.lstsq(A_submx, b, rcond=None)
        x[inhomog_col] = -1  # put back in the -1 coefficient for b
    return x
