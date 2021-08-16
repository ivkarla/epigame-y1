# ca: connectivity analysis

from numpy import correlate, average, array, angle, mean, sign, exp, zeros, abs, unwrap
from numpy.linalg import norm
from awc import error
from scipy.signal import coherence, hilbert, csd
from matplotlib.mlab import cohere
from itertools import combinations
from data_legacy import butter_filter
from sklearn.preprocessing import normalize

def connectivity_analysis(epochs, method, dtail=False, *opts):
    print('Connectivity Analysis: '+str(method).split()[1])
    result = [] 
    for i,e in enumerate(epochs):    
        mat = zeros((len(e),len(e)))                                    
        nid, pairs = list(range(len(e))), []
        for a in range(len(nid)):                             
            if dtail:
                for b in range(len(nid)): pairs.append((a,b))
            else:
                for b in range(a, len(nid)): pairs.append((a,b))                                       
        for pair in pairs:                                                       
            mat[pair[0],pair[1]] = method(e[pair[0]], e[pair[1]], *opts)
        result.append(mat)     
        print('{}: completed '.format(i), end='\n')                                                                                       
    return result

def phaselock(signal1, signal2):
    '''Phase Locking Value (PLV) of two signals, signal1 and signal2
       Returns the phase locking value, that comprehends the range [0,1]
       Signals that are more synchronized in phase are more closer to 1. 
       
       NOTE: To properly obtain the PLV of two signals, they have to be bandpass filtered beforehand!!
    '''
    sig1_hil = hilbert(signal1)                           #Hilbert transformations -> obtains the analytic signal (complex number)
    sig2_hil = hilbert(signal2)
    phase1 = angle(sig1_hil)                           #Argument of the complex number, instant phase angle
    phase2 = angle(sig2_hil)
    phase_dif = phase1-phase2                             #Phases Difference
    plv = abs(mean(exp(complex(0,1)*phase_dif)))    #Phase Locking value
    return plv

def phaselag(signal1, signal2):
    '''Phase Lag Index (PLI) of two signals, signal1 and signal2
       Returns the phase locking value, that comprehends the range [0,1]
       A PLI of zero indicates either no coupling or coupling with a phase difference centered around 0 mod π.
       A PLI of 1 indicates perfect phase locking at a value of Δϕ different from 0 mod π. 
       The stronger this nonzero phase locking is, the larger PLI will be. Note that PLI does no longer indicate,
       which of the two signals is leading in phase. Whenever needed, however, this information can be easily recovered, 
       for instance, by omitting the absolute value when computing PLI. 
    '''
    sig1_hil = hilbert(signal1)                 #Hilbert transformations -> obtains the analytic signal (complex number)
    sig2_hil = hilbert(signal2)
    phase1 = angle(sig1_hil)                 #Argument of the complex number, instant phase angle
    phase2 = angle(sig2_hil)
    phase_dif = phase1-phase2                   #Phases Difference
    pli = abs(mean(sign(phase_dif)))      # Phase Lag Index
    return pli

def spectral_coherence(signal1, signal2, fs, imag=False):
    Pxy = csd(signal1,signal2,fs=fs, scaling='spectrum')[1] # cross power spectral density (csd), Pxy, using Welch’s method.
    Pxx = csd(signal1,signal1,fs=fs, scaling='spectrum')[1]
    Pyy = csd(signal2,signal2,fs=fs, scaling='spectrum')[1]
    
    if imag: return average((Pxy.imag)**2/(Pxx*Pyy))     # imaginary coherence
    elif not imag: return average(abs(Pxy)**2/(Pxx*Pyy)) # coherence

def cross_correlation(signal1, signal2):                                                    
    return correlate(signal1, signal2, mode="valid")

def PEC(nse,n):
    print('{}: '.format(n), end='')
    return array(error(nse, 2)[1])

def PAC(signal1,signal2,fs):
    """ Phase-amplitude coupling """   
    low = butter_filter(signal1,1,4,fs) #delta
    high = butter_filter(signal2,30,70,fs) #low gamma
    
    low_hil = hilbert(low)
    low_phase_angle = unwrap(angle(low_hil)) #instantaneous phase angle  
    high_env_hil = hilbert(abs(hilbert(high))) #envelope extraction
    high_phase_angle = unwrap(angle(high_env_hil))
    phase_dif = low_phase_angle - high_phase_angle #phases Difference
    plv = abs(mean(exp(complex(0,1)*phase_dif))) #phase locking value
    return plv
    
    
    