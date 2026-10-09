import numpy as np
from scipy.optimize import branq


def eigval_P(a,b,rho):
    """Compute the theoretical eigenvalues of the matrix P."""
    thetap = (a+b)/2 + np.sqrt((a-b)**2/4 + a*b*rho**2)
    thetam = (a+b)/2 - np.sqrt((a-b)**2/4 + a*b*rho**2)
    # lam = (a+b)/2 +- sqrt((a-b)^2 /4 + ab rho^2)
    return thetap, thetam

def eigvect_P(a,b,x1,x2,rho):

    thetap, thetam = eigval_P(a,b,rho)

    rp = (thetap-a)/(a*rho)
    vp = x1 + rp * x2
    vp /= np.linalg.norm(vp)

    rm = (thetam-a)/(a*rho)
    vm = x1 + rm * x2
    vm /= np.linalg.norm(vm)

    return vp, vm

def eigval_Y(a,b,rho, c=0):
    """ Top eig of Y from BBP """
    th= eigval_P(a,b,rho)
    return [theta + 1/np.maximum(theta, 1e-12) +c if theta > 1 else 2
        for theta in th]

def overlap_Y_P(theta):
    theta = np.asarray(theta)
    return np.where(
        theta > 1 +1e-12,
        np.sqrt(1 - 1 / theta**2),
        0
    )

def overlap_naive_signal(theta, rho, a, b):
    r = rho*a /(theta - a)
    N = np.sqrt(r**2 + 2*rho *r + 1)
    c1 = r/N
    ov = overlap_Y_P(theta)
    o1 = ov*theta*abs(c1)/a
    o2 = ov*theta*abs(2)/b
    return o1, o2

def beta_value(mu, lambda1, m1, lambda2, rho):  # builds the beta matrix, diagonalizes it, returns the top eigenvalue
    """Top eigenvalue of the 3x3 matrix beta.

    beta is the low-rank part of A = Y2 - I + 2 mu F1(x),  x = m1 x1 + sqrt(1 - m1^2) g_perp,
    i.e.  lambda2 x2 x2^T + 2 mu (lambda1 - 1)^2 x x^T  (1/N terms of F1 dropped),
    written in the orthonormal basis ( x1, (x2 - rho x1)/sqrt(1 - rho^2), g_perp ):

        beta11 = 2 mu (lambda1-1)^2 m1^2 + lambda2 rho^2
        beta22 = (1 - rho^2) lambda2
        beta33 = 2 mu (lambda1-1)^2 (1 - m1^2)
        beta12 = beta21 = lambda2 rho sqrt(1 - rho^2)
        beta13 = beta31 = 2 mu (lambda1-1)^2 m1 sqrt(1 - m1^2)
        beta23 = beta32 = 0

    Note: the top eigenvalue of A itself is then (BBP) theta + 1/theta - 1 if theta > 1,
    else the bulk edge 2 - 1 = 1, with theta = beta_value(...).
    """
    c = 2 * mu * (lambda1 - 1) ** 2
    s_rho = np.sqrt(max(0.0, 1 - rho ** 2))
    s_m = np.sqrt(max(0.0, 1 - m1 ** 2))
    beta = np.array([
        [c * m1 ** 2 + lambda2 * rho ** 2, lambda2 * rho * s_rho,    c * m1 * s_m],
        [lambda2 * rho * s_rho,            lambda2 * (1 - rho ** 2), 0.0],
        [c * m1 * s_m,                     0.0,                      c * (1 - m1 ** 2)],
    ])
    return np.linalg.eigvalsh(beta)[-1]