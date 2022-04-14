import numpy as np
from scipy.stats import gamma
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

class KCI:
 
    def kernel(self, x, y):
        T = len(y)

        x = np.array(x)
        y = np.array(y)
        x = x - np.mean(x)
        x = x / np.std(x)
        y = y - np.mean(y)
        y = y / np.std(y)

        if T < 200:
            width = 0.8
        elif T < 1200:
            width = 0.5
        else:
            width = 0.3

        theta = 1 / (width**2)

        Kx = 1.0 * RBF(theta).__call__(x, x)
        Ky = 1.0 * RBF(theta).__call__(y, y)

        return Kx, Ky
    
    def eigdec(self, mat, num_eig):
        
        eig_K, eiv = np.linalg.eig((mat + np.transpose(mat)) / 2)
        idx = np.argsort(eig_K)[-num_eig:]
        idx = idx[::-1]
        
        eig_K_sort = np.transpose(np.matrix([eig_K[i] for i in idx]))
        eiv_sort = np.transpose(np.matrix([eiv[i] for i in idx]))
        
        return eig_K_sort, eiv_sort
    
    def bootstrap(self, x, y):
        
        T = len(y)
        num_eig = int(np.floor(T / 2))
        T_BS = 1000
        thresh = 1e-6
        
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        
        Kx, Ky = self.kernel(x, y)
        eig_Kx, eivx = self.eigdec(Kx, num_eig)
        eig_Ky, eivy = self.eigdec(Ky, num_eig)
        
        eig_prod = np.matmul(np.matmul(eig_Kx, np.ones((1, num_eig))), np.matmul(np.ones((num_eig, 1)), np.transpose(eig_Ky)))
        eig_prod = np.ravel(eig_prod)
        
        eig_prod_II = np.matrix([x for x in eig_prod if x > (max(eig_prod) * thresh)])
        
        if (len(eig_prod_II) * T) < 1e6:
            f_rand1 = np.random.chisquare(1, (len(eig_prod_II), T_BS))
            null_dist = np.matmul(np.transpose(eig_prod_II) / T, f_rand1)
            
        else:
            null_dist = np.zeros((1, T_BS))
            Length = np.maximum(np.floor(1e6 / T), 100)
            Itmax = np.floor(len(eig_prod_II)/Length)
            
            for i in np.arange(Itmax):
                f_rand1 = np.random.chisquare(1, (Length, T_BS))
                null_dist = null_dist + np.matmul(np.transpose(eig_prod_II[(i - 1) * Length:i*Length]) / T, f_rand1)
        
        null_dist = np.sort(null_dist)
        
        stat = self.statistic(x, y)
        pvalue = len([x for x in null_dist if x.any() > stat]) / T_BS
        
        return pvalue

    def statistic(self, x, y):

        T = len(y)
        
        H = np.eye(T) - np.ones((T, T)) / T

        Kx, Ky = self.kernel(x, y)

        Kx = np.matmul(np.matmul(H, Kx), H)
        Ky = np.matmul(np.matmul(H, Ky), H)

        stat = np.trace(np.matmul(Kx, Ky))

        return stat

    def test(self, x, y, method):

        if method == 'Gamma':
            T = len(y)
            Kx, Ky = self.kernel(x, y)
            stat = self.statistic(x, y)

            mean_appr = (np.trace(Kx) * np.trace(Ky)) / T
            var_appr = 2 * np.trace(np.matmul(Kx, Kx)) * np.trace(np.matmul(Ky, Ky)) / T**2
            k_appr = mean_appr**2 / var_appr
            theta_appr = var_appr / mean_appr
            pvalue = 1 - np.mean(gamma.cdf(stat, k_appr, theta_appr))

            self.stat = stat

        if method == 'bootstrap':
            stat = self.statistic(x, y)
            pvalue = bootstrap(x, y)
            self.stat = stat

        return stat, pvalue