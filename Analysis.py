import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib import cm
from scipy.signal import argrelextrema

plt.style.use('ggplot')


## reading data saved in .csv file
t_vec,ir_vec,red_vec = [],[],[]
with open('test_data.csv',newline='') as csvfile:
    csvreader = csv.reader(csvfile,delimiter=',')
    for row in csvreader:
        t_vec.append(float(row[0]))
        ir_vec.append(float(row[1]))
        red_vec.append(float(row[2]))
        print("Appending")
       

s1 = 0 # change this for different range of data
s2 = len(t_vec) # change this for ending range of data
t_vec = np.array(t_vec[s1:s2])
ir_vec = ir_vec[s1:s2]
red_vec = red_vec[s1:s2]

# sample rate and heart rate ranges
samp_rate = 1/np.mean(np.diff(t_vec)) # average sample rate for determining peaks
heart_rate_range = [0,250] # BPM
heart_rate_range_hz = np.divide(heart_rate_range,60.0)
max_time_bw_samps = 1/heart_rate_range_hz[1] # max seconds between beats
max_pts_bw_samps = max_time_bw_samps*samp_rate # max points between beats

heart_rate_span = [10,250] # max span of heart rate
smoothing_size = 20 # convolution smoothing size
pts = len(t_vec) # points used for peak finding (400 Hz, I recommend at least 4s (1600 pts)

## plotting time series data
fig = plt.figure(figsize=(14,8))
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time [s]',fontsize=24)
ax1.set_ylabel('IR Amplitude',fontsize=24,color='#CE445D',labelpad=10)
ax1.tick_params(axis='both',which='major',labelsize=16)
plt1 = ax1.plot(t_vec,ir_vec,label='IR',color='#CE445D',linewidth=4)
ax1_2 = plt.twinx()
#ax1_2.grid('off')
ax1_2.set_ylabel('Red Amplitude',fontsize=24,color='#37A490',labelpad=10)
ax1_2.tick_params(axis='y',which='major',labelsize=16)
plt2 = ax1_2.plot(t_vec,red_vec,label='Red',color='#37A490',linewidth=4)
lns = plt1+plt2
labels = [l.get_label() for l in lns]
ax1_2.legend(lns,labels,fontsize=16,loc='upper center')
plt.xlim([t_vec[0],t_vec[-1]])
plt.tight_layout(pad=1.2)
#plt.savefig('max30102_python_example.png',dpi=300,facecolor=[252/255,252/255,252/255])
#plt.show()
print("Working")

## FFT and plotting frequency spectrum of data
f_vec = np.arange(0,int(len(t_vec)/2))*(samp_rate/(len(t_vec)))
f_vec = f_vec*60
print(len(f_vec))
fft_var = np.fft.fft(red_vec)
#fft_values = 2*np.abs(fft_var[1:int(len(fft_var)/2)])
#fft_axis = np.abs(int(len(fft_var)/2))
#fft_zero = 0*np.abs(fft_var)
fft_var = 2*np.abs(fft_var[1:int(len(fft_var)/2)+1])
print(len(fft_var))
# print(np.append(np.abs(fft_var[0]),1,1))
#fft_var = np.append(fft_values,fft_zero,1)
#fft_var = np.append(np.abs(fft_var[1]),2*np.abs(fft_var[1:int(len(fft_var)/2)]),np.abs(fft_var[200]))
#fft_var = fft_values
#int(len(fft_var)/2)

bpm_max_loc = np.argmin(np.abs(f_vec-heart_rate_range[1]))
f_step = 1
f_max_loc = np.argmax(fft_var[f_step:bpm_max_loc])+f_step
#print('BPM: {0:2.1f}'.format(f_vec[f_max_loc]))
#fig2 = plt.figure(figsize=(14,8))
#ax2 = fig2.add_subplot(111)
#ax2.loglog(f_vec,fft_var,color=[50/255,108/255,136/255],linewidth=4)
#color=[50/255,108/255,136/255],linewidth=4
#ax2.set_xlim([0,f_vec[-1]])
#ax2.set_ylim([np.min(fft_var)-np.std(fft_var),np.max(fft_var)])
#ax2.tick_params(axis='both',which='major',labelsize=16)
#ax2.set_xlabel('Frequency [BPM]',fontsize=24)
#ax2.set_ylabel('Amplitude',fontsize=24)
#ax2.annotate('Heart Rate: {0:2.0f} BPM'.format(f_vec[f_max_loc]),
#             xy = (f_vec[f_max_loc],fft_var[f_max_loc]+(np.std(fft_var)/10)),xytext=(-10,70),
#            textcoords='offset points',arrowprops=dict(facecolor='k'),
#             fontsize=16,horizontalalignment='center')
# fig2.savefig('max30102_fft_heart_rate.png',dpi=300,facecolor=[252/255,252/255,252/255])
#plt.show()
#plt.ion()

fig3 = plt.figure(figsize=(14,8))
ax3 = fig3.add_subplot(111)
line1, = ax3.plot(np.arange(0,pts),np.zeros((pts,)),linewidth=4,label='Smoothed Data')
line2, = ax3.plot(0,0,label='Gradient Peaks',marker='o',linestyle='',color='k',markersize=10)
ax3.set_xlabel('Time [s]',fontsize=16)
ax3.set_ylabel('Amplitude',fontsize=16)
ax3.legend(fontsize=16)
ax3.tick_params(axis='both',which='major',labelsize=16)
#plt.show()
# calculating heart rate
#t1 = time.time()
y_vals = red_vec
samp_rate = 1/np.mean(np.diff(t_vec)) # average sample rate for determining peaks
min_time_bw_samps = (60.0/heart_rate_span[1])

# convolve, calculate gradient, and remove bad endpoints
y_vals = np.convolve(y_vals,np.ones((smoothing_size,)),'same')/smoothing_size
red_grad = np.gradient(y_vals,t_vec)
red_grad[0:int(smoothing_size/2)+1] = np.zeros((int(smoothing_size/2)+1,))
red_grad[-int(smoothing_size/2)-1:] = np.zeros((int(smoothing_size/2)+1,))  

y_vals = np.append(np.repeat(y_vals[int(smoothing_size/2)],int(smoothing_size/2)),y_vals[int(smoothing_size/2):-int(smoothing_size/2)])
y_vals = np.append(y_vals,np.repeat(y_vals[-int(smoothing_size/2)],int(smoothing_size/2)))

# update plot with new Time and red/IR data
line1.set_xdata(t_vec)
line1.set_ydata(y_vals)

ax3.set_xlim([np.min(t_vec),np.max(t_vec)])
if line1.axes.get_ylim()[0]<0.95*np.min(y_vals) or\
   np.max(y_vals)>line1.axes.get_ylim()[1] or\
   np.min(y_vals)<line1.axes.get_ylim()[0]:
                ax3.set_ylim([np.min(y_vals),np.max(y_vals)])

# peak locator algorithm
peak_locs = np.where(red_grad<-np.std(red_grad))
#if len(peak_locs[0])==0:
    #continue

prev_pk = peak_locs[0][0]
true_peak_locs,pk_loc_span = [],[]
for ii in peak_locs[0]:
    y_pk = y_vals[ii]
    if (t_vec[ii]-t_vec[prev_pk])<min_time_bw_samps:
        pk_loc_span.append(ii)
    else:
        if pk_loc_span==[]:
            true_peak_locs.append(ii)
        else:
            true_peak_locs.append(int(np.mean(pk_loc_span)))
            pk_loc_span = []

    prev_pk = int(ii)

t_peaks = [t_vec[kk] for kk in true_peak_locs]
#if t_peaks==[]:
    #continue
#else:
print('BPM: {0:2.1f}'.format(60.0/np.mean(np.diff(t_peaks))))
ax3.set_title('{0:2.0f} BPM (RED)'.format(60.0/np.mean(np.diff(t_peaks))),fontsize=24)
# plot gradient peaks on original plot to view how BPM is calculated
scatter_x,scatter_y = [],[]
for jj in true_peak_locs:
    scatter_x.append(t_vec[jj])
    scatter_y.append(y_vals[jj])
   
line2.set_data(scatter_x,scatter_y)

fig4 = plt.figure(figsize=(14,8))
ax4 = fig4.add_subplot(111)
line3, = ax4.plot(np.arange(0,pts),np.zeros((pts,)),linewidth=4,label='Smoothed Data')
line4, = ax4.plot(0,0,label='Gradient Peaks',marker='o',linestyle='',color='k',markersize=10)
ax4.set_xlabel('Time [s]',fontsize=16)
ax4.set_ylabel('Amplitude',fontsize=16)
ax4.legend(fontsize=16)
ax4.tick_params(axis='both',which='major',labelsize=16)
#plt.show()
# calculating heart rate
#t1 = time.time()
y_vals1 = ir_vec
#samp_rate1 = 1/np.mean(np.diff(t_vec)) # average sample rate for determining peaks
#min_time_bw_samps1 = (60.0/heart_rate_span[1])

# convolve, calculate gradient, and remove bad endpoints
y_vals1 = np.convolve(y_vals1,np.ones((smoothing_size,)),'same')/smoothing_size
ir_grad = np.gradient(y_vals1,t_vec)
ir_grad[0:int(smoothing_size/2)+1] = np.zeros((int(smoothing_size/2)+1,))
ir_grad[-int(smoothing_size/2)-1:] = np.zeros((int(smoothing_size/2)+1,))  

y_vals1 = np.append(np.repeat(y_vals1[int(smoothing_size/2)],int(smoothing_size/2)),y_vals1[int(smoothing_size/2):-int(smoothing_size/2)])
y_vals1 = np.append(y_vals1,np.repeat(y_vals1[-int(smoothing_size/2)],int(smoothing_size/2)))

# update plot with new Time and red/IR data
line3.set_xdata(t_vec)
line3.set_ydata(y_vals1)

ax4.set_xlim([np.min(t_vec),np.max(t_vec)])
if line3.axes.get_ylim()[0]<0.95*np.min(y_vals1) or\
   np.max(y_vals1)>line3.axes.get_ylim()[1] or\
   np.min(y_vals1)<line3.axes.get_ylim()[0]:
                ax4.set_ylim([np.min(y_vals1),np.max(y_vals1)])

#print(np.max(y_vals1))
# peak locator algorithm
peak_locs1 = np.where(ir_grad<-np.std(ir_grad))
#if len(peak_locs[0])==0:
    #continue

prev_pk1 = peak_locs1[0][0]
true_peak_locs1,pk_loc_span1 = [],[]
for ee in peak_locs1[0]:
    y_pk1 = y_vals1[ii]
    if (t_vec[ee]-t_vec[prev_pk1])<min_time_bw_samps:
        pk_loc_span1.append(ee)
    else:
        if pk_loc_span1==[]:
            true_peak_locs1.append(ee)
        else:
            true_peak_locs1.append(int(np.mean(pk_loc_span1)))
            pk_loc_span1 = []

    prev_pk1 = int(ee)

t_peaks1 = [t_vec[tt] for tt in true_peak_locs1]
#if t_peaks==[]:
    #continue
#else:
print('BPM: {0:2.1f}'.format(60.0/np.mean(np.diff(t_peaks1))))
ax4.set_title('{0:2.0f} BPM (IR)'.format(60.0/np.mean(np.diff(t_peaks1))),fontsize=24)
# plot gradient peaks on original plot to view how BPM is calculated
scatter_x1,scatter_y1 = [],[]
for rr in true_peak_locs1:
    scatter_x1.append(t_vec[rr])
    scatter_y1.append(y_vals1[rr])
   
line4.set_data(scatter_x1,scatter_y1)

#max_r = np.argmax(red_vec)
#min_r = np.argmin(red_vec)
a = y_vals1
b = y_vals1
c = y_vals
d = y_vals
max_ir = argrelextrema(y_vals1, np.greater)
min_ir = argrelextrema(y_vals1, np.less)
max_r = argrelextrema(y_vals, np.greater)
min_r = argrelextrema(y_vals, np.less)
#max_r = argrelextrema(red_vec, np.greater)
#min_r = argrelextrema(red_vec, np.less)
rmax = a[max_ir]
rmin = b[min_ir]
irmax = c[max_r]
irmin = d[min_r]
#print(r)
#print(y_vals1)
#print(min_r)

check_length = np.array([len(rmax), len(rmin), len(irmax), len(irmin)])
#spo2 = np.append(spo2, 1)
#spo2 = np.append(spo2, 2)
#print(spo2)
#print(check_length)
ac_ir = np.array([irmax[0]-irmin[0]])
dc_ir = np.array([(irmax[0] - ac_ir[0])/2])
ac_r = np.array([rmax[0]-rmin[0]])
dc_r = np.array([(rmax[0] - ac_r[0])/2])
for ij in range(1,check_length.min()):
    ac_ir = np.append(ac_ir, irmax[ij]-irmin[ij])
    dc_ir = np.append(dc_ir, (irmax[ij] - ac_ir[ij])/2)
    ac_r = np.append(ac_r, rmax[ij]-rmin[ij])
    dc_r = np.append(dc_r, (rmax[ij] - ac_r[ij])/2)

big_R = np.array([(ac_r[0]/dc_r[0])/(ac_ir[0]/dc_ir[0])])
for ji in range(1,check_length.min()):
    big_R = np.append(big_R, (ac_r[ji]/dc_r[ji])/(ac_ir[ji]/dc_ir[ji]))
   
a1 = -16.666666
b1 = 8.333333
c1 = 100.0
spo2 = np.array([a1*(big_R[0]*big_R[0]) + b1*(big_R[0]) + c1])
for df in range(1,check_length.min()):
    spo2 = np.append(spo2, a1*(big_R[df]*big_R[df]) + b1*(big_R[df]) + c1)

spo2clone = np.array([100.0])
anomaly = np.array([1])
for rc in range(0,len(spo2)):
    if spo2[rc] <= 115 and spo2[rc] >= 70:
        spo2clone = np.append(spo2clone, spo2[rc])
    else:
        anomaly = np.append(anomaly, spo2[rc])

e = spo2clone
min_spo2 = argrelextrema(spo2clone, np.less)
minsforspo2 = e[min_spo2]

time = np.array([0])
peak_detector = np.array([0])
for ty in range(1,len(spo2clone)):
    time = np.append(time,ty)
    peak_detector = np.append(peak_detector,0)
   
for tu in range(1,len(spo2clone)):
    for ti in range(1,len(minsforspo2)):
        if spo2clone[tu] == minsforspo2[ti]:
            plt.plot(time[tu],minsforspo2[ti],'o')


#drop = 0;
#drop_detect = 0;
#timer = 0;
#for tg in range(0,len(spo2clone)-1):
#   if spo2clone[tg] >= spo2clone[tg+1]+1 and tg >= timer:
#        drop = drop + 1;
#        timer = tg + 30;
#    else if spo2clone[tg] >= spo2clone[tg+1]+1:
       
#    else
       
       
print('Anomalies in Spo2')            
print(anomaly)
print('Number of Anomalies in Spo2')
print(len(anomaly))
print('BPM average')
print(((60.0/np.mean(np.diff(t_peaks1))) + (60.0/np.mean(np.diff(t_peaks))))/2)
print('Spo2 Average')
print(np.mean(spo2clone))
fig5 = plt.figure(figsize=(14,8))
ax5 = fig5.add_subplot(111)
ax5.set_xlabel('Time [s]',fontsize=24)
ax5.set_ylabel('Spo2',fontsize=24,color='#CE445D',labelpad=10)
plt5 = ax5.plot(time, spo2clone,label='Spo2',color='#CE445D',linewidth=4)


for tu in range(0,len(spo2clone)):
    for ti in range(0,len(minsforspo2)):
        if spo2clone[tu] == minsforspo2[ti] and minsforspo2[ti] <= 98:
            ax5.plot(time[tu],minsforspo2[ti],'o')
            peak_detector[tu] = minsforspo2[ti]
           
drop = 0
instance = 0
for tp in range(0,len(peak_detector)):
    if peak_detector[tp] > 0 and drop == 0:
        drop = 30;
        instance = instance + 1;
    elif peak_detector[tp] > 0:
        drop = 30;
    if drop > 0:
        drop = drop - 1;

print('Number of Drop Instances')
print(instance)
#approximately 400hz
number_of_hours = len(time)/1440000
AHI = instances/number_of_hours
if AHI > 30.0:
    print('Severe OSA')
elif AHI > 15.0
    print('Moderate OSA')
elif AHI > 5
    print('Mild OSA')
else
    print('No OSA')


#spo2deriv = np.gradient(spo2clone)
#print(spo2deriv)

#fig6 = plt.figure(figsize=(14,8))
#ax6 = fig6.add_subplot(111)
#ax6.set_xlabel('Peaks in order',fontsize=24)
#ax6.set_ylabel('HR_IR',fontsize=24,color='#CE445D',labelpad=10)
#plt6 = ax6.plot(abs(spo2deriv),label='Spo2',color='#CE445D',linewidth=4)

#fig7 = plt.figure(figsize=(14,8))
#ax7 = fig7.add_subplot(111)
#ax7.set_xlabel('Time [s]',fontsize=24)
#ax7.set_ylabel('HR_R',fontsize=24,color='#CE445D',labelpad=10)
#plt7 = ax7.plot(np.diff(t_peaks),label='Spo2',color='#CE445D',linewidth=4)
#plt5.xlim([spo2.min(),spo2.max()])
#fig3 = plt.figure(figsize=(14,8))
#ax3 = fig.add_subplot(111)
#ax3.set_xlabel('Time [s]',fontsize=24)
#ax3.set_ylabel('IR Amplitude',fontsize=24,color='#CE445D',labelpad=10)
#ax3.tick_params(axis='both',which='major',labelsize=16)
#plt3 = ax1.plot(t_vec,y_vals,label='IR',color='#CE445D',linewidth=4)
#ax3_2 = plt.twinx()
#ax3_2.grid('off')
#ax3_2.set_ylabel('Red Amplitude',fontsize=24,color='#37A490',labelpad=10)
#ax3_2.tick_params(axis='y',which='major',labelsize=16)
#plt4 = ax3_2.plot(t_vec,red_vec,label='Red',color='#37A490',linewidth=4)
#lns = plt1+plt2
#labels = [l.get_label() for l in lns]
#ax3_2.legend(lns,labels,fontsize=16,loc='upper center')
#plt.xlim([t_vec[0],t_vec[-1]])
#plt.tight_layout(pad=1.2)

plt.show()
