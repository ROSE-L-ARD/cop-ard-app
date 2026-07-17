from itertools import combinations_with_replacement
from typing import Tuple

import numpy as np
import scipy.optimize
from astropy import stats as ast
from cop_ard_core.logger import logger


def cost_function_l2(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    L2 cost function

    Parameters:
    - params: array: parameters of the model
    - X: array: points where the model is evaluated
    - y: array: values of the points where the model is evaluated

    Returns:
    - cost: array: values of the cost function
    """
    assert X.ndim == 2
    assert X.shape[1] == params.shape[0]
    assert X.shape[0] == y.shape[0]

    # Return the cost function
    return np.sum((y - fit(X, params)) ** 2)


def cost_function_l1(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    L1 cost function

    Parameters:
    - params: array: parameters of the model
    - X: array: model matrix
    - y: array: values of the points where the model is evaluated

    Returns:
    - cost: array: values of the cost function
    """
    assert X.ndim == 2
    assert X.shape[1] == params.shape[0]
    assert X.shape[0] == y.shape[0]

    # Return the cost function
    return np.sum(np.abs(y - fit(X, params)))


def fit(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit the model

    Parameters:
    - X: array: model matrix
    - params: array: parameters of the model

    Returns:
    - y: array: values of the points where the model is evaluated
    """
    assert X.ndim == 2
    assert X.shape[1] == params.shape[0]

    # Return the model
    return X.dot(params)


def get_model_matrix(x: np.ndarray, regions: np.ndarray | None = None, degree: int = 1) -> np.ndarray:
    """
    Compute the model matrix as polinomial function with linear combnation of the positions x

    Parameters:
    - x: array: points where the model is evaluated
    - degree: int: degree of the model

    Returns:
    - X: array: model matrix
    """
    assert x.ndim == 2
    assert x.shape[1] > 0

    # Initialization
    if regions is not None:
        n_regions = regions.max() + 1
        X = np.zeros((x.shape[0], n_regions), dtype="float64")
        for k in range(n_regions):
            X[regions == k, k] = 1
    else:
        n_regions = 1
        X = np.ones((x.shape[0], n_regions), dtype="float64")

    # Different degrees
    for k in np.arange(1, degree + 1):
        # Define all combinations
        for ll in combinations_with_replacement(np.arange(x.shape[1]), k):
            # Construct the model matrix
            X = np.hstack((np.prod(x[:, list(ll)], axis=1).reshape(-1, 1), X))

    return X


def estimate_global_surface(
    x: np.ndarray,
    y: np.ndarray,
    regions: np.ndarray | None = None,
    degree: int = 0,
    norm: int = 2,
    method: str | None = None,
    sigma: float = 3.0,
    range: Tuple[float, float] = (-np.inf, np.inf),
    convergency_threshold: float = 1e-6,
    maximum_iterations: int = 100,
    **kwargs,
):
    """
    Remove large scale spatial trend (n-dimensional polynomial model) from the data

    Parameters:
    - x: array: points where the model is evaluated
    - y: array: values of the points where the model is evaluated
    - degree: int: degree of the model
    - norm: int: Norm considered in minimization (1 or 2)
    - method: str | None: minimization method
    - debug: bool = False: print debug information
    - return_fit = False: bool = False, return the fitted model in correspondence of each input point
    - return_residues: bool = False, return the residues of the fit in correspondence of each input point
    - sigma: float = 3., threshold to discard outliers
    - ranges: Tuple[float, float] | None, minimum and maximum accepted values for optimization problem

    Returns:
    - output: array: values of the points where the model is evaluated
    - res: array: if return_residues is enabled
    - y_fit: array: if return_fit is enabled
    """

    assert x.ndim == 2
    assert x.shape[1] > 0
    assert degree >= 0

    # Set minimization norm and method
    if norm == 1:
        cost_function = cost_function_l1
        method_opt = "Nelder-Mead" if (method is None) else method
    elif norm == 2:
        cost_function = cost_function_l2
        method_opt = "BFGS" if (method is None) else method
    else:
        raise ValueError(f"Norm {norm} not supported")

    # Construct model incidence matrix
    X = get_model_matrix(x, regions=regions, degree=degree)

    # Init solution (null)
    sol_prec = np.zeros(X.shape[1], dtype="float64")

    # Iterations
    for i in np.arange(maximum_iterations):
        # Remove outliers
        y_prec = fit(X, sol_prec)
        diff = y - y_prec

        mask = (
            (~ast.sigma_clip(diff, sigma=sigma, maxiters=1, stdfunc=ast.mad_std).mask)  # ty: ignore[unresolved-attribute]  mask is present as masked=True by default
            & (diff > range[0])
            & (diff < range[1])
        )

        # Call minimization on selected points
        output = scipy.optimize.minimize(cost_function, sol_prec, args=(X[mask, :], y[mask]), method=method_opt)

        # new surface fit
        y_new = fit(X, output.x)

        # compute difference wrt previous iteration
        diff_prec = np.max(np.abs(y_new - y_prec))
        logger.debug(f"iter {i}: {diff_prec=} {np.max(np.abs(diff))=}")

        # check convergence
        if diff_prec < convergency_threshold:
            break
        else:
            sol_prec = output.x

    logger.info(f"Convergency reached in {i} iterations (Solution: {output.x})")

    return output


def fit_2d_poly(x: np.ndarray, y: np.ndarray, data: np.ndarray, degree: int = 1, norm: int = 1):

    # estimate only in correspondence of finite values
    mask = np.isfinite(data)

    # select points
    data_sel = data[mask]
    xx, yy = x.ravel()[mask], y.ravel()[mask]

    # Axis normalization
    x_ = np.vstack([xx, yy]).T
    x_0 = x_.mean(axis=0)
    x_ = x_ - x_0
    x_sc = np.abs(x_).max(axis=0)
    x_ = x_ / x_sc

    return estimate_global_surface(x_, data_sel, regions=None, degree=degree, norm=norm), x_sc, x_0
