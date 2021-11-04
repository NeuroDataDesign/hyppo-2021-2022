import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

T = np.array([10,50,100,500,1000])

dcorr_samp10 = 2.18
dcorr_samp50 = 1.18
dcorr_samp100 = 1.83
dcorr_samp500 = 33.04
dcorr_samp1000 = 162.44
dcorr_hyppo = [dcorr_samp10,dcorr_samp50,dcorr_samp100,dcorr_samp500,dcorr_samp1000]

dcorr_samp101 = 1.48
dcorr_samp501 =  3.74
dcorr_samp1001 = 7.56
dcorr_samp5001 = 105.93
dcorr_samp10001 = 557
dcorr_hyppo1 = [dcorr_samp101,dcorr_samp501,dcorr_samp1001,dcorr_samp5001,dcorr_samp10001]

dcorr_samp10tot = 7.99
dcorr_samp50tot = 6.81
dcorr_samp100tot = 8
dcorr_samp500tot = 42.52
dcorr_samp1000tot = 188.41
dcorr_dcorr = [dcorr_samp10tot,dcorr_samp50tot,dcorr_samp100tot,dcorr_samp500tot,dcorr_samp1000tot]

dcorr_samp10tot1 = 1.15
dcorr_samp50tot1 = 3.24
dcorr_samp100tot1 = 6.5
dcorr_samp500tot1 = 110
dcorr_samp1000tot1 = 511.06
dcorr_dcorr1 = [dcorr_samp10tot1,dcorr_samp50tot1,dcorr_samp100tot1,dcorr_samp500tot1,dcorr_samp1000tot1]

# 300 represents number of points to make between T.min and T.max
xnew = np.linspace(T.min(), T.max(), 300) 

spl_hyppo = make_interp_spline(T, dcorr_hyppo, k=3)  # type: BSpline
power_smoothhyppo = spl_hyppo(xnew)

spl_hyppo1 = make_interp_spline(T, dcorr_hyppo1, k=3)  # type: BSpline
power_smoothhyppo1 = spl_hyppo1(xnew)

spl_dcorr = make_interp_spline(T, dcorr_dcorr, k=3)  # type: BSpline
power_smoothdcorr = spl_dcorr(xnew)

spl_dcorr1 = make_interp_spline(T, dcorr_dcorr1, k=3)  # type: BSpline
power_smoothdcorr1 = spl_dcorr1(xnew)

plt.plot(xnew, power_smoothhyppo,'b',label = 'Hyppo mgc w/ Workers = -1')
plt.plot(xnew, power_smoothhyppo1,'r',label = 'Hyppo mgc w/ Workers = 1')
plt.plot(xnew, power_smoothdcorr,'g',label = 'Scipy mgc w/ Workers = -1')
plt.plot(xnew, power_smoothdcorr1,'k',label = 'Scipy mgc w/ Workers = -1')
plt.legend()
plt.xlabel('Number of Samples')
plt.ylabel('Time Ran (seconds)')
plt.title('Comparison of Time for Hyppo and Scipy mgc')
plt.show()