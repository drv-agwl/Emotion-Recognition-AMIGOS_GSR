import os,csv
from gsr_utils import *
from gsr_feature import *
import warnings
from sys import argv	
import math

def getlabel(filaname):
	label_all = {}
	readfile = open(filaname,'r')
	for n, row in enumerate(list(csv.reader(readfile))):
		participant = str(int(n/16)+1)
		exp = str(int(n%16)+1)
		label_all[participant+'_'+exp] = ([float(i) for i in row])
	readfile.close()
	return label_all

def readfile(filename):
	signal = np.genfromtxt(filename,delimiter=',')
	eeg_signal=signal[:,:14] 
	ecg_signal=signal[:,15:16]
	gsr_signal=signal[:,-1]
	return eeg_signal, ecg_signal, gsr_signal

def writesignal(filename, participant ,signal):
	file = open(filename, 'a', encoding = 'big5')
	file.write(participant)
	for i in range(len(signal)):
		file.write(','+str(signal[i]))
	file.write('\n')
	file.close()

def feature_set(data,sig_type,fs):
	print('    get statistics_feature...')
	f1, name1 = statistics_feature(data)
	f2, name2 = statistics_feature(np.diff(data))
	f3, name3 = statistics_feature(np.diff(np.diff(data)))
	name2 = rename(name2, '1df.')
	name3= rename(name3, '2df.')
	print('    get freq_feature...')
	f4, name4 = freq_feature(data,sig_type,fs)
	print('    get DWT_feature...')
	f5, name5 = DWT(data)
	print('    get entropy_feature...')
	f6, name6 = entropy_feature(data,match=2,scale=20)
	f = f1+f2+f3+f4+f5+f6
	name = name1+name2+name3+name4+name5+name6
	name = rename(name, sig_type)
	return f, name

def main():
	writefile = argv[1]
	label = getlabel('data/label.csv') 
	labelname = ['arousal','valence','dominance','liking','familiarity','neutral','disgust','happiness','surprise','anger','fear','sadness']
	writename = False
	for participant in range(1,41):
		if participant==int(8):
			continue
		for exp in range(1,17):

			filename = str(participant)+'_'+str(exp)
			print('data:'+filename)
			eeg_signal, ecg_signal, gsr_signal_r = readfile('data/' + filename + '.csv')
			if participant==int(17) and exp == int(9):
				gsr_signal_r = gsr_signal_r[6000:14181]	
			else:
				gsr_signal_r = gsr_signal_r[-8181:]
			#gsr_nopeak = delpeak(gsr_signal_r,3,2)
			gsr_signal = 1e6/gsr_signal_r
			gsr_filter = low_pass_filter(gsr_signal, fc=2, fs=128, order=5)
			#plotsignal([gsr_filter])
			
			'''
			N = 960
			nfs = math.ceil(128/len(gsr_filter)*N)
			SC = downsample(gsr_filter,fs=128,nfs=nfs)
			SC = SC[int((len(SC)-N)/2):int((len(SC)-N)/2)+N]
			'''
			nfs = 16
			SC = downsample(gsr_filter,fs=128,nfs=nfs)
			df = np.diff(SC)
			size = int(1. * 16)
			df = tools.smoother(df,'bartlett',size)['signal']
			zeros, = tools.zero_cross(signal=df, detrend=False)
			if len(zeros)==0:
				gsr_filter = detrend(gsr_filter)+np.min(gsr_filter)
				SC = detrend(SC)+np.min(SC)

			feature_size = []
			all_feature = []
			all_name = []
			for task in ['ori.','det.','win.','df.','bd.h.','bd.m.','bd.l.','lo.h.','lo.l.','CDA.','CVX.']:
			#for task in ['CVX.']:
				print('get '+task+'feature...')
				if task == 'ori.':
					feature, name = feature_set(gsr_filter, 'o.', fs=128)
				elif task == 'CDA.':
					onset, peak, amp, driver, phasic = SCR_generate(gsr_filter,fs=nfs,min_amplitude=0.1,task=task)	
					tonic = SC - phasic
					#plotpeak(SC,driver,tonic,onset,peak,amp,filename,nfs)
					f1, name1 = SCR_feature(driver,onset,peak,amp,fs=nfs)
					print('  get phasic...')
					f2, name2 = feature_set(phasic,'p.',fs=nfs)
					print('  get tonic...')
					f3, name3 = feature_set(tonic,'t.',fs=nfs)
					feature = f1+f2+f3
					name = name1+name2+name3
					name = rename(name,task)
				elif task == 'CVX.':
					onset, peak, amp, phasic, tonic = SCR_generate(SC,fs=nfs,min_amplitude=0.1,task=task)	
					#plotpeak(SC,phasic,tonic,onset,peak,amp,filename,nfs)
					f1, name1 = SCR_feature(phasic,onset,peak,amp,fs=nfs)
					print('  get phasic...')
					f2, name2 = feature_set(phasic, 'p.',fs=nfs)
					print('  get tonic...')
					f3, name3 = feature_set(tonic,'t.',fs=nfs)
					feature = f1+f2+f3
					name = name1+name2+name3
					name = rename(name,task)
				else:
					onset, peak, amp, phasic = SCR_generate(SC,fs=nfs,min_amplitude=0.1,task=task)
					tonic = SC - phasic
					#plotpeak(SC,phasic,tonic,onset,peak,amp,filename,nfs)	
					f1, name1 = SCR_feature(phasic,onset,peak,amp,fs=nfs)
					print('  get phasic...')
					f2, name2 = feature_set(phasic, 'p.',fs=nfs)
					print('  get tonic...')
					f3, name3 = feature_set(tonic,'t.',fs=nfs)
					feature = f1+f2+f3
					name = name1+name2+name3
					name = rename(name,task)
				if len(feature_size)<11:
					feature_size.append(len(feature))
				all_feature += feature
				all_name += name				
			print(feature_size,sum(feature_size))					
			all_feature = all_feature+label[filename]
			all_name = all_name+labelname
			if writename == False:
				writesignal(writefile,'', all_name)			
				writename = True	
				file = open('feature_all.csv', 'w')
				for i,n in enumerate(all_name):
					file.write(str(n)+'\n')
				file.close()
				writename = True
			writesignal(writefile, filename, all_feature)
			print('==========')
if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	main()