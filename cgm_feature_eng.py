import datetime as Datetime
import pandas
import statistics
import numpy, csv

import matplotlib.pyplot as plt

import numpy.polynomial.polynomial as poly
from numpy.polynomial import Polynomial
import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks
from datetime import datetime
import heapq


from sklearn.decomposition import IncrementalPCA
import pylab as pl
from sklearn import datasets
from sklearn.decomposition import PCA

path = "D:\ASU Data\CSE 572\DataFolder\\"

cgmLp1 = pandas.read_csv(path+"CGMSeriesLunchPat1.csv")
cgmLp2 = pandas.read_csv(path+"CGMSeriesLunchPat2.csv")
cgmLp3 = pandas.read_csv(path+"CGMSeriesLunchPat3.csv")
cgmLp4 = pandas.read_csv(path+"CGMSeriesLunchPat4.csv")
cgmLp5 = pandas.read_csv(path+"CGMSeriesLunchPat5.csv")

cgmLpdt1 = pandas.read_csv(path+"CGMDatenumLunchPat1.csv")
cgmLpdt2 = pandas.read_csv(path+"CGMDatenumLunchPat2.csv")
cgmLpdt3 = pandas.read_csv(path+"CGMDatenumLunchPat3.csv")
cgmLpdt4 = pandas.read_csv(path+"CGMDatenumLunchPat4.csv")
cgmLpdt5 = pandas.read_csv(path+"CGMDatenumLunchPat5.csv")


#print(cgmLp1.iloc[28], type(cgmLp1), numpy.std(cgmLp1.iloc[28]), numpy.mean(cgmLp1.iloc[28]))
#print(cgmLp1.iloc[28].values)
#print(cgmLp1.iloc[28].values[0:30])
#print(numpy.where(numpy.isnan(cgmLp1.iloc[28].values), 0, cgmLp1.iloc[28].values))

#print("STD:", statistics.stdev(numpy.where(numpy.isnan(cgmLp1.iloc[28].values), 0, cgmLp1.iloc[28].values)))
# By replacing the nan with zero the mean and std are getting affected significantly so can not do this

# Now replace by Mean
#print(numpy.where(numpy.isnan(cgmLp1.iloc[28].values), numpy.mean(cgmLp1.iloc[28]), cgmLp1.iloc[28].values))

#print("STD:", statistics.stdev(numpy.where(numpy.isnan(cgmLp1.iloc[28].values), numpy.mean(cgmLp1.iloc[28]), cgmLp1.iloc[28].values)))
# By replacing the nan with Mean The Mean and STD are not getting affected so we will use this

#Feature 1: Coefficient of variation {STD,Mean}

stdarr = []
meanarr = []
CV = []

for i in range(0,len(cgmLp1)):
    #print(cgmLp1.iloc[i])
    stdarr.append(numpy.std(cgmLp1.iloc[i]))
    meanarr.append(numpy.mean(cgmLp1.iloc[i]))
    tCV = numpy.std(cgmLp1.iloc[i]) / numpy.mean(cgmLp1.iloc[i])
    #print(tCV)
    CV.append(tCV)

plt.title("Patient 1 All events Data")
plt.subplot(131)
plt.plot(stdarr, label="std")
plt.legend(loc='best')
plt.subplot(132)
plt.plot(meanarr, label="mean")
plt.legend(loc='best')
plt.subplot(133)
plt.plot(CV, label="Coefficient of variation")
plt.legend(loc='best')
plt.show()

#print(cgmLpdt1.columns)

#print(cgmLpdt1.iloc[2][:1])
#print(cgmLpdt1.iloc[2])

print(cgmLpdt1.iloc[2].values[0])
print(pandas.to_datetime(cgmLpdt1.iloc[2].values[0]-719529,unit='D'))
print(pandas.to_datetime(cgmLpdt1.iloc[2].values[1]-719529,unit='D'))

#print(datetime.fromtimestamp(cgmLpdt1.iloc[2].values[0]))
#print(datetime.fromtimestamp(cgmLpdt1.iloc[2].values[1]))
#print(datetime.fromordinal(int(str(cgmLpdt1.iloc[2].values[1]).split(".")[0])))
#print(datetime.fromtimestamp(cgmLpdt1.iloc[2][:2]))
#for idx in cgmLp1.index:
#    print("Row:", cgmLp1[idx])


#Feature 2:	Polynomial Curve fitting the CGM Data {polynomial factors}

cgmLpdt1_ = cgmLpdt1.applymap(lambda x : pandas.to_datetime(x-719529 , unit = 'D'))

#Convert timestamp to the human readable dates and time
print(cgmLpdt1_.iloc[0].values)


#Ref: https://numpy.org/doc/1.18/reference/generated/numpy.polynomial.polynomial.polyfit.html#numpy.polynomial.polynomial.polyfit
#c, stats = numpy.polynomial.polynomial.polyfit(cgmLpdt1_.iloc[0].values, cgmLp1.iloc[0].values, 4, full=True)

#time array for calculations
timearr = numpy.arange(0,151,5)

print(timearr)

#c, stats = numpy.polynomial.polynomial.polyfit(cgmLpdt1.iloc[0].values, cgmLp1.iloc[0].values, 4, full=True)

c, stats = numpy.polynomial.polynomial.polyfit(timearr, cgmLp1.iloc[0].values[::-1], 5, full=True)

#ffit = numpy.polynomial(c[::-1],x_new)

ffit = numpy.polyval(timearr,c)
#plt.plot(timearr,ffit)

#plt.show()
p = Polynomial.fit(timearr, numpy.where(numpy.isnan(cgmLp1.iloc[2].values), numpy.mean(cgmLp1.iloc[2]), cgmLp1.iloc[2].values)[::-1], 7)

#replacing the nan by mean and reversing
print(numpy.where(numpy.isnan(cgmLp1.iloc[2].values), numpy.mean(cgmLp1.iloc[2]), cgmLp1.iloc[2].values)[::-1])

#plt.subplot(133)
print("Actual:", cgmLp1.iloc[2].values[::-1])
#plt.plot(timearr,cgmLp1.iloc[1].values[::-1], label='Actual')

plt.plot(timearr,numpy.where(numpy.isnan(cgmLp1.iloc[2].values), numpy.mean(cgmLp1.iloc[2]), cgmLp1.iloc[2].values)[::-1], label='Actual')
plt.plot(*p.linspace(), label='Poly fit')
plt.legend(loc='best')
plt.title('Polynomial Curve fitting for patient 1 LE 3')
plt.xlabel('Time values')
plt.ylabel('CGM levels')

plt.show()

# Coefficient are the features in this case we are considering 5 coefficients

print("Coefficients:", c)
print("stats:", stats)


#Feature 3: Discrete Fourier Transform Amplitudes


DFTarr = np.fft.fft(numpy.where(numpy.isnan(cgmLp1.iloc[3].values), numpy.mean(cgmLp1.iloc[3]), cgmLp1.iloc[3].values)[::-1])

DFTarr_abs = numpy.absolute(DFTarr)
print(DFTarr)
print(DFTarr_abs)
#plt.plot(timearr, DFTarr_abs , label = "CGM Data")
# First term is DC term has to be ignored https://dsp.stackexchange.com/questions/12972/discrete-fourier-transform-what-is-the-dc-term-really
plt.plot( DFTarr_abs[1:31/2] , label = "DFT plot")
#plt.plot(timearr, numpy.where(numpy.isnan(cgmLp1.iloc[3].values), numpy.mean(cgmLp1.iloc[3]), cgmLp1.iloc[3].values)[::-1] , label = "DFT")
plt.legend(loc='best')
plt.show()


# finding peaks  https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy

# Scipy fft


print("scipyfft:", scipy.fftpack.rfft(numpy.where(numpy.isnan(cgmLp1.iloc[3].values), numpy.mean(cgmLp1.iloc[3]), cgmLp1.iloc[3].values)[::-1],n=5))

# Above scipy fft values are our features; we are ignoring the first value as its a dc values and doesn't contain any important information
# 4th Feature: Peaks of CGM Velocity peaks

CGMVelocity  = numpy.diff(numpy.where(numpy.isnan(cgmLp1.iloc[1].values), numpy.mean(cgmLp1.iloc[1]), cgmLp1.iloc[1].values)[::-1],n=1)
print("CGMVelocity:", CGMVelocity)
plt.plot(CGMVelocity, label="CGM Velocity")
plt.legend(loc='best')
plt.show()

peaksarridx = scipy.signal.find_peaks(CGMVelocity)
print("ArrayPeaksIndex:", peaksarridx[0])
peakArray = CGMVelocity[peaksarridx[0]]
print("PeakArray:", numpy.take(CGMVelocity, peaksarridx[0]))
print("PeakArray:", CGMVelocity[peaksarridx[0]])
print ("Top 3 Peaks from CGM Velocity:", peakArray[heapq.nlargest(3, range(len(peakArray)), peakArray.take)])



###############  Part 2: ############### (b):
# Theory explanation





###############  Part 3: ############### (c):
# Feature 1:

#Patient 1

stdarr = []
meanarr = []
CV = []

for i in range(0,len(cgmLp1)):
    #print(cgmLp1.iloc[i])
    stdarr.append(numpy.std(cgmLp1.iloc[i]))
    meanarr.append(numpy.mean(cgmLp1.iloc[i]))
    tCV = numpy.std(cgmLp1.iloc[i]) / numpy.mean(cgmLp1.iloc[i])
    #print(tCV)
    CV.append(tCV)


plt.subplot(131)
plt.title('Patient 1 All events Data')
plt.plot(stdarr, label="std")
plt.legend(loc='best')
plt.subplot(132)
plt.plot(meanarr, label="mean")
plt.legend(loc='best')
plt.subplot(133)
plt.plot(CV, label="Coefficient of variation")
plt.legend(loc='best')
plt.show()


#Patient 2

stdarr = []
meanarr = []
CV = []

for i in range(0,len(cgmLp2)):
    #print(cgmLp1.iloc[i])
    stdarr.append(numpy.std(cgmLp2.iloc[i]))
    meanarr.append(numpy.mean(cgmLp2.iloc[i]))
    tCV = numpy.std(cgmLp2.iloc[i]) / numpy.mean(cgmLp2.iloc[i])
    #print(tCV)
    CV.append(tCV)


plt.subplot(131)
plt.title('Patient 2 All events Data')
plt.plot(stdarr, label="std")
plt.legend(loc='best')
plt.subplot(132)
plt.plot(meanarr, label="mean")
plt.legend(loc='best')
plt.subplot(133)
plt.plot(CV, label="Coefficient of variation")
plt.legend(loc='best')
plt.show()

#Patient 3

stdarr = []
meanarr = []
CV = []

for i in range(0,len(cgmLp3)):
    #print(cgmLp1.iloc[i])
    stdarr.append(numpy.std(cgmLp3.iloc[i]))
    meanarr.append(numpy.mean(cgmLp3.iloc[i]))
    tCV = numpy.std(cgmLp3.iloc[i]) / numpy.mean(cgmLp3.iloc[i])
    #print(tCV)
    CV.append(tCV)


plt.subplot(131)
plt.title('Patient 3 All events Data')
plt.plot(stdarr, label="std")
plt.legend(loc='best')
plt.subplot(132)
plt.plot(meanarr, label="mean")
plt.legend(loc='best')
plt.subplot(133)
plt.plot(CV, label="Coefficient of variation")
plt.legend(loc='best')
plt.show()


#Patient 4

stdarr = []
meanarr = []
CV = []

for i in range(0,len(cgmLp4)):
    #print(cgmLp1.iloc[i])
    stdarr.append(numpy.std(cgmLp4.iloc[i]))
    meanarr.append(numpy.mean(cgmLp4.iloc[i]))
    tCV = numpy.std(cgmLp4.iloc[i]) / numpy.mean(cgmLp4.iloc[i])
    #print(tCV)
    CV.append(tCV)


plt.subplot(131)
plt.title('Patient 4 All events Data')
plt.plot(stdarr, label="std")
plt.legend(loc='best')
plt.subplot(132)
plt.plot(meanarr, label="mean")
plt.legend(loc='best')
plt.subplot(133)
plt.plot(CV, label="Coefficient of variation")
plt.legend(loc='best')
plt.show()

#Patient 5

stdarr = []
meanarr = []
CV = []

for i in range(0,len(cgmLp5)):
    #print(cgmLp1.iloc[i])
    stdarr.append(numpy.std(cgmLp5.iloc[i]))
    meanarr.append(numpy.mean(cgmLp5.iloc[i]))
    tCV = numpy.std(cgmLp5.iloc[i]) / numpy.mean(cgmLp5.iloc[i])
    #print(tCV)
    CV.append(tCV)


plt.subplot(131)
plt.title('Patient 5 All events Data')
plt.plot(stdarr, label="std")
plt.legend(loc='best')
plt.subplot(132)
plt.plot(meanarr, label="mean")
plt.legend(loc='best')
plt.subplot(133)
plt.plot(CV, label="Coefficient of variation")
plt.legend(loc='best')
plt.show()



# Feature 2: Discrete Fourier Transform Amplitudes


#c, stats = numpy.polynomial.polynomial.polyfit(cgmLpdt1.iloc[0].values, cgmLp1.iloc[0].values, 4, full=True)

c, stats = numpy.polynomial.polynomial.polyfit(timearr, numpy.where(numpy.isnan(cgmLp1.iloc[0].values), numpy.mean(cgmLp1.iloc[0]), cgmLp1.iloc[0].values)[::-1], 7, full=True)

#ffit = numpy.polynomial(c[::-1],x_new)

ffit = numpy.polyval(timearr,c)
#plt.plot(timearr,ffit)

#plt.show()
p = Polynomial.fit(timearr, numpy.where(numpy.isnan(cgmLp1.iloc[0].values), numpy.mean(cgmLp1.iloc[0]), cgmLp1.iloc[0].values)[::-1], 7)

#replacing the nan by mean and reversing
print(numpy.where(numpy.isnan(cgmLp1.iloc[0].values), numpy.mean(cgmLp1.iloc[0]), cgmLp1.iloc[0].values)[::-1])

#plt.subplot(133)
print("Actual:", cgmLp1.iloc[0].values[::-1])
#plt.plot(timearr,cgmLp1.iloc[1].values[::-1], label='Actual')

plt.plot(timearr,numpy.where(numpy.isnan(cgmLp1.iloc[0].values), numpy.mean(cgmLp1.iloc[0]), cgmLp1.iloc[0].values)[::-1], label='Actual')
plt.plot(*p.linspace(), label='Poly fit')
plt.legend(loc='best')
plt.title('Polynomial Curve fitting for patient 1 LE 2')
plt.xlabel('Time values')
plt.ylabel('CGM levels')

plt.show()

# Coefficient are the features in this case we are considering 5 coefficients

print("F2Coefficients:", c)
print("F2stats:", stats)


coefficient = []
fit_stats = []

for i in range(0,len(cgmLp1)):
    OrdarrLEp1 = numpy.where(numpy.isnan(cgmLp1.iloc[i].values), numpy.mean(cgmLp1.iloc[i]), cgmLp1.iloc[i].values)[::-1]
    c, stats = numpy.polynomial.polynomial.polyfit(timearr, OrdarrLEp1, 7, full=True)
    coefficient.append(c)
    fit_stats.append(stats)

print("Cofficienct patient 1:", coefficient)
print("fit_stats patient 1:", fit_stats[1][1])

rms = []
for i in range(0,len(fit_stats)): rms.append(fit_stats[i][0])
plt.plot(rms)
plt.show()



coefficientp2 = []
fit_statsp2 = []

for i in range(0,len(cgmLp2)):
    OrdarrLEp2 = numpy.where(numpy.isnan(cgmLp2.iloc[i].values), numpy.mean(cgmLp2.iloc[i]), cgmLp2.iloc[i].values)[::-1]
    c, stats = numpy.polynomial.polynomial.polyfit(timearr, OrdarrLEp2, 7, full=True)
    coefficientp2.append(c)
    fit_statsp2.append(stats)

print("Cofficienct patient 1:", coefficientp2)
print("fit_stats patient 1:", fit_statsp2[1][1])


rmsp2 = []
for i in range(0,len(fit_statsp2)): rmsp2.append(fit_statsp2[i][0])

plt.plot(rmsp2)

# The range of goodness value can be argued


# Feature 3: fft values
# We decompose the curve into parts of curves

print("scipyfft:", scipy.fftpack.rfft(numpy.where(numpy.isnan(cgmLp1.iloc[3].values), numpy.mean(cgmLp1.iloc[3]), cgmLp1.iloc[3].values)[::-1],n=5))

CoeffP1 = []
for i in range(0,len(cgmLp1)):
    OrdarrLEp1 = numpy.where(numpy.isnan(cgmLp1.iloc[i].values), numpy.mean(cgmLp1.iloc[i]), cgmLp1.iloc[i].values)[::-1]
    Coeff = scipy.fftpack.rfft(OrdarrLEp1, n=5)
    CoeffP1.append(Coeff)

print(CoeffP1)

CoeffP2 = []
for i in range(0,len(cgmLp2)):
    OrdarrLEp2 = numpy.where(numpy.isnan(cgmLp2.iloc[i].values), numpy.mean(cgmLp2.iloc[i]), cgmLp2.iloc[i].values)[::-1]
    Coeff = scipy.fftpack.rfft(OrdarrLEp2, n=5)
    CoeffP2.append(Coeff)

print(CoeffP2)


# Feature 4: "Top 3 Peaks from CGM Velocity:"


CGMVelocity = numpy.diff(numpy.where(numpy.isnan(cgmLp1.iloc[1].values), numpy.mean(cgmLp1.iloc[1]), cgmLp1.iloc[1].values)[::-1],n=1)
print("CGMVelocity:", CGMVelocity)
plt.plot(CGMVelocity, label="CGM Velocity")
plt.legend(loc='best')
plt.show()

peaksarridx = scipy.signal.find_peaks(CGMVelocity)
print("ArrayPeaksIndex:", peaksarridx[0])
peakArray = CGMVelocity[peaksarridx[0]]
print("PeakArray:", numpy.take(CGMVelocity, peaksarridx[0]))
print("PeakArray:", CGMVelocity[peaksarridx[0]])
print ("Top 3 Peaks from CGM Velocity:", peakArray[heapq.nlargest(3, range(len(peakArray)), peakArray.take)])


#Patient 1
Top3PeaksP1 = []

for i in range(0,len(cgmLp1)):
    OrdarrLEp1 = numpy.where(numpy.isnan(cgmLp1.iloc[i].values), numpy.mean(cgmLp1.iloc[i]), cgmLp1.iloc[i].values)[::-1]
    CGMVelocity = numpy.diff(OrdarrLEp1, n=1)
    peaksarridx = scipy.signal.find_peaks(CGMVelocity)
    peakArray = CGMVelocity[peaksarridx[0]]
    Top3Peaks = peakArray[heapq.nlargest(3, range(len(peakArray)), peakArray.take)]
    Top3PeaksP1.append(Top3Peaks)

print("Top3PeaksP1:", Top3PeaksP1)

Top3PeaksP2 = []

for i in range(0,len(cgmLp2)):
    OrdarrLEp2 = numpy.where(numpy.isnan(cgmLp2.iloc[i].values), numpy.mean(cgmLp2.iloc[i]), cgmLp2.iloc[i].values)[::-1]
    CGMVelocity = numpy.diff(OrdarrLEp2, n=1)
    peaksarridx = scipy.signal.find_peaks(CGMVelocity)
    peakArray = CGMVelocity[peaksarridx[0]]
    Top3Peaks = peakArray[heapq.nlargest(3, range(len(peakArray)), peakArray.take)]
    Top3PeaksP2.append(Top3Peaks)

print("Top3PeaksP1:", Top3PeaksP2)



###############  Part 4: ############### (d): Feature matrix

feature_matrix = []

cgmLpAll = cgmLp1.append(cgmLp2).append(cgmLp3).append(cgmLp4).append(cgmLp5)


stdarr = []
meanarr = []
CVAll = []

for i in range(0,len(cgmLpAll)):
    #print(cgmLp1.iloc[i])
    stdarr.append(numpy.std(cgmLpAll.iloc[i]))
    meanarr.append(numpy.mean(cgmLpAll.iloc[i]))
    tCV = numpy.std(cgmLpAll.iloc[i]) / numpy.mean(cgmLpAll.iloc[i])
    #print(tCV)
    CVAll.append(tCV)

print(CVAll)

#CVAll = np.asarray(CVAll)

CVmat = np.asmatrix(np.asarray(CVAll).reshape(216L,1))

coefficientp = []
fit_statsp = []

timearrAll = numpy.arange(0,206,5)

for i in range(0,len(cgmLpAll)):
    OrdarrLEp = numpy.where(numpy.isnan(cgmLpAll.iloc[i].values), numpy.mean(cgmLpAll.iloc[i]), cgmLpAll.iloc[i].values)[::-1]
    c, stats = numpy.polynomial.polynomial.polyfit(timearrAll, OrdarrLEp, 7, full=True)
    coefficientp.append(c)
    fit_statsp.append(stats)

print("Cofficienct patient:", coefficientp)

print("fit_stats patient:", fit_statsp[1][1])
Polymat = numpy.asmatrix(coefficientp)

CoeffPAll = []
for i in range(0,len(cgmLpAll)):
    OrdarrLEp = numpy.where(numpy.isnan(cgmLpAll.iloc[i].values), numpy.mean(cgmLpAll.iloc[i]), cgmLpAll.iloc[i].values)[::-1]
    Coeff = scipy.fftpack.rfft(OrdarrLEp, n=5)
    CoeffPAll.append(Coeff)

print(CoeffPAll)

fftmat = numpy.asmatrix(CoeffPAll)

Top3PeaksPAll = []

for i in range(0,len(cgmLpAll)):
    OrdarrLEp = numpy.where(numpy.isnan(cgmLpAll.iloc[i].values), numpy.mean(cgmLpAll.iloc[i]), cgmLpAll.iloc[i].values)[::-1]
    CGMVelocity = numpy.diff(OrdarrLEp, n=1)
    peaksarridx = scipy.signal.find_peaks(CGMVelocity)
    peakArray = CGMVelocity[peaksarridx[0]]
    if (len(peakArray) == 0):
       Top3Peaks = np.array([0,0,0])
    else:
        Top3Peaks = peakArray[heapq.nlargest(3, range(len(peakArray)), peakArray.take)].reshape(3L,)
        # print(peakArray, len(peakArray))
    Top3PeaksPAll.append(Top3Peaks)

print("Top3PeaksP1:", Top3PeaksPAll)

PeakVelocitymat = numpy.asmatrix(Top3PeaksPAll)

feature_matrix = np.concatenate((CVmat, Polymat,fftmat,PeakVelocitymat), axis=1)
feature_matrix = np.nan_to_num(feature_matrix)
print(feature_matrix, "shape of feature matrix:", feature_matrix.shape)


###############  Part 5: ############### (e): PCA

#Using the PCA to generate the new PCA matrix
pca = PCA(n_components=5)
pca_reduced_new_mat = pca.fit_transform(feature_matrix)

plt.plot(pca_reduced_new_mat[:,0])
plt.xlabel('Meal events across the data')
plt.ylabel('PCA component1 ');
plt.show()
plt.plot(pca_reduced_new_mat[:,1])
plt.xlabel('Meal events across the data')
plt.ylabel('PCA component2 ');
plt.show()

plt.plot(pca_reduced_new_mat[:,2])
plt.xlabel('Meal events across the data')
plt.ylabel('PCA component3 ');
plt.show()

plt.plot(pca_reduced_new_mat[:,3])
plt.xlabel('Meal events across the data')
plt.ylabel('PCA component4 ');
plt.show()

plt.plot(pca_reduced_new_mat[:,4])
plt.xlabel('Meal events across the data')
plt.ylabel('PCA component5 ');
plt.show()

print(pca_reduced_new_mat, "shape:", pca_reduced_new_mat.shape)

###############  Part 6: ############### (f): Top 5 PCA


print pca.explained_variance_ratio_


print(sum(pca.explained_variance_ratio_))

#pca = PCA().fit(feature_matrix.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()


