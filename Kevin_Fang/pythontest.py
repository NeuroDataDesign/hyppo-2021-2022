import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

T = np.array([10,50,100,500,1000])

dcorr_samp10 = (sum([1.59,1.43,1.44]))/3
dcorr_samp50 = (sum([1.46,1.49,1.51]))/3
dcorr_samp100 = (sum([1.49,1.50,1.48]))/3
dcorr_samp500 = (sum([3.61,3.75,3.69]))/3
dcorr_samp1000 = (sum([16.19,15.65,15.45]))/3
dcorr_hyppo = [dcorr_samp10,dcorr_samp50,dcorr_samp100,dcorr_samp500,dcorr_samp1000]

dcorr_samp101 = (sum([0.365,0.336,0.320])/3)
dcorr_samp501 = (sum([0.400,0.412,0.375]))/3
dcorr_samp1001 = (sum([0.580,0.546,0.563]))/3
dcorr_samp5001 = (sum([8.48,8.25,8.49]))/3
dcorr_samp10001 = (sum([68.624,65.40,64.81]))/3
dcorr_hyppo1 = [dcorr_samp101,dcorr_samp501,dcorr_samp1001,dcorr_samp5001,dcorr_samp10001]

dcorr_samp10tot = (sum([3.77,3.836,3.721]))/3
dcorr_samp50tot = (sum([3.713,3.595,3.895]))/3
dcorr_samp100tot = (sum([3.559,3.919,3.988]))/3
dcorr_samp500tot = (sum([5.787,5.778,5.358]))/3
dcorr_samp1000tot = (sum([20.912,20.546,20.285]))/3
dcorr_dcorr = [dcorr_samp10tot,dcorr_samp50tot,dcorr_samp100tot,dcorr_samp500tot,dcorr_samp1000tot]

dcorr_samp10tot1 = (sum([0.263,0.257,0.270]))/3
dcorr_samp50tot1 = (sum([0.333,0.326,0.326]))/3
dcorr_samp100tot1 = (sum([0.504,0.522,0.532]))/3
dcorr_samp500tot1 = (sum([14.431,13.647,13.393]))/3
dcorr_samp1000tot1 = (sum([58.937,56.169,58.447]))/3
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

plt.plot(xnew, power_smoothhyppo,'b',label = 'Hyppo dcorr w/ Workers = -1')
plt.plot(xnew, power_smoothhyppo1,'r',label = 'Hyppo dcorr w/ Workers = 1')
plt.plot(xnew, power_smoothdcorr,'g',label = 'Scipy dcorr w/ Workers = -1')
plt.plot(xnew, power_smoothdcorr1,'k',label = 'Scipy dcorr w/ Workers = -1')
plt.legend()
plt.xlabel('Number of Samples')
plt.ylabel('Time Ran (seconds)')
plt.title('Comparison of Time for Hyppo and Scipy Dcorr')
plt.show()