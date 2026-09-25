
import numpy as np

class TwoS_theory:
    def __init__(self, N, lam1, lam2, rho, mu=1.0, alpha=np.sqrt(0.5)):
            self.N = N
            self.lam1 = lam1
            self.lam2 = lam2
            self.rho = rho
            self.alpha = alpha
            self.mu = mu        #methode A et B

            #analyse theorique
            self.a = alpha * lam1
            self.b = np.sqrt(1 - alpha ** 2) * lam2

            self.eig_val = {"P": eigval_P(self.a, self.b, self.rho), 
                            "naive": self.eigval_Y(), 
                            "A": None, 
                            "B": None}
            
            self.eig_vect = {"P": eigvect_P(self.a, self.b, self.x1, self.x2, self.rho), 
                             "naive": None, 
                             "A": None, 
                             "B": None}

            #signals and observations
            self.Y1, self.Y2, self.x1, self.x2 = self.two_correlated_spikes()
    
            #noisless signal P
            self.P = self.alpha *self.lam1 * np.outer(self.x1, self.x1) + np.sqrt(1 - self.alpha ** 2) *self.lam2 * np.outer(self.x2, self.x2)

#### get
    def get_eigP(self):
        return self.eig_val["P"], self.eig_vect["P"]

    def get_eigY(self):
            return self.eig_val["naive"], self.eig_vect["naive"]

### generative function 
    def two_correlated_spikes(self):
        "corraleted signals xi and observations Yi"
        cov = [[1, self.rho], [self.rho, 1]]
        X = np.random.multivariate_normal([0, 0], cov, self.N)
        x1 = X[:, 0]
        x2 = X[:, 1]
        x1 = x1 / np.linalg.norm(x1)
        x2 = x2 / np.linalg.norm(x2)
        Y1 = self.lam1 * np.outer(x1, x1) + self.symmetric_gaussian_matrix()
        Y2 = self.lam2 * np.outer(x2, x2) + self.symmetric_gaussian_matrix()
        return Y1, Y2, x1, x2

    def symmetric_gaussian_matrix(self):
        "GOE(N)"
        n = self.N
        G = np.random.normal(0, 1, (n, n))
        return (G + G.T) / np.sqrt(2 * n)


#### Theory formulas or class
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

def eigval_Y(thetap, thetam):
    """ Top eig of Y from BBP """
    th= [thetap, thetam]
    return [theta + 1/np.maximum(theta, 1e-12) if theta > 1 else 2
        for theta in th]


def overlap_Y_P(theta):
    
    if theta > 1e-12:
        return np.sqrt(1 - 1/(theta**2))
    else:
        return 100

def overlap_naive_signal(theta, rho, a, b):
    
    r = rho*a /(theta - a)
    N = np.sqrt(r**2 + 2*rho *r + 1)
    c1 = r/N
    ov = overlap_Y_P(theta)
    o1 = ov*theta*abs(c1)/a
    o2 = ov*theta*abs(2)/b
    return o1, o2



