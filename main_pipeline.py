from eeg import EEG, SET, STEp, epoch, secs, ms, np, struct
from core import REc as rec
from data_legacy import notch, dwindle, record, band, upsample
from connectivity import connectivity_analysis, phaselock, phaselag, spectral_coherence, PEC, cross_correlation, PAC

from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from itertools import combinations
from sys import setrecursionlimit
import os

def preprocess(eeg, epoch, limit=500): 
    sampling, rse = limit, epoch
    if eeg.fs == limit: rse = epoch
    elif eeg.fs == 250: rse = upsample(epoch, eeg.fs, limit) if eeg.fs<limit else dwindle(epoch, int(eeg.fs/limit)-1) 
    else: rse = upsample(epoch, eeg.fs, limit) if eeg.fs<limit else dwindle(epoch, int(eeg.fs/limit)-2) 
    nse = notch(rse, fs=sampling, order=2)
    return nse

def analyze(set, nodes, kratio=.1, random_state=31, **mopts):
    X, Y = np.array([np.array((record(x).include(*nodes).T).include(*nodes)).flatten() for x in set.X]), set.y
    model, scaler, k = SVC, StandardScaler, int(round(len(Y)*kratio))
    if random_state == None: random_state = np.random.randint(0xFFFFFFFF)
    C = Pipeline([('scaler', scaler()), ('model', model(**mopts))])
    cv = KFold(k, shuffle=True, random_state=random_state)
    cvs = cross_val_score(C, X, Y, cv=cv)
    return cvs

def minimax_of(results):
    return max(results)*(min(results)/np.average(results)) 

def enlist(nodes, labels, results, symbol='<->', fx=minimax_of):
    tag = symbol.join([labels[n] for n in nodes])
    return (nodes, tag, results, fx(results))

def check_until(net, set=1, fall=0, at=-1):
    while set < len(net):
        score = np.average([n[at] for n in net[:set]])
        if score>=fall:
            fall=score
            set+=1
        else: break
    return set

source = os.cwd()+"../data/raw/" 
preprocessed = os.cwd()+"../data/preprocessed/"
out = os.cwd()+"../data/results/"

""" Initialization of variables: method and frequency band of choice """  
window = input("Time window:\n 1. Non-seizure (baseline)\n 2. Pre-seizure\n 3. Transition to seizure\n 4. Seizure\n Indicate a number: ")
method_idx = input("Connectivity method:\n 1. PEC\n 2. Spectral Coherence\n 3. Phase Lock Value\n 4. Phase-Lag Index\n 5. Cross-correlation\n 6. Phase-amplitude coupling\n Indicate a number: ")
ext = ""
if "2" == method_idx: 
    im = input("Imaginary part (Y/N): ").upper()
    if im == "Y": imag,ext = True,"I"
    elif im == "N": imag,ext = False,"R"
bands, Bands = input("Filter the signal: Y/N ").upper(), False
if bands=="N": bands = "w"
elif bands=="Y": 
    Bands = True
    mn = int(input("Set band range min: "))
    mx = int(input("Set band range max: "))
    bands = (mn,mx)
dict_methods = {'1':PEC, '2':spectral_coherence, '3':phaselock, '4':phaselag, '5':cross_correlation, '6':PAC}
method_codes = {'1':"PEC", '2':"SC_", '3':"PLV", '4':"PLI", '5':"CC", '6':"PAC"}   
files = list(os.listdir(source))
subjects = [fname[0:3] for fname in files]
step, span = 500, 1000            #to increase stats power, decrease step to 250 or 100, modyfing their ratio accordingly (span/step)
ratio = span/step
A, B = 'S', 'E'                   #S and E labels both mean non-seizure, kept from the original version for convenience
items = 30                        #number of epochs to analyse
retry_ratio = 0                   #at the beginning I though to use a percentage of all found networks, but that leads to an exponential increase that prooves to be unmanageable
max_netn = 10                     #max number of nodes per network. This 10 is to avoid too long processing times and also a statistical failure, if networks are too big they will ultimately cover all the explored area
    
for subid in subjects:   
    print('processing {}'.format(subid), end='...')
    name = ''
    for filename in files:
        if filename.startswith(subid): name = filename; break        
    eeg = EEG.from_file(source+name, epoch(ms(step), ms(span)))
    print(subid, "\n sampling: ", eeg.fs)
    
    SET(eeg, _as='N')
    SET(eeg, 'EEG inicio', 'S')
    SET(eeg, 'EEG fin', 'E', epoch.END)
    eeg.optimize()
    eeg.remap()
    units = int((eeg.notes['EEG fin'][0].time - eeg.notes['EEG inicio'][0].time)*ratio)
    
    if window == "1":
        pre = int(round(units))
        eeg.tag(('S','E'), S=range(-pre,0,1), E=range(0,-units,-1))
    elif window == "2":
        pre = int(round(units*.6)) 
        eeg.tag(('S', 'E'), S=range(-pre,0,1), E=range(0,-units,-1))
    elif window == "3" or window == "4":
        pre = int(round(units*.3))
        eeg.tag(('S', 'E'), S=range(-pre,pre,1), E=range(0,-units,-1)) 

    a, ai = eeg.sample.get(A, items)  # a -> epochs with 'N' ; ai -> index of these epochs 
    b, bi = eeg.sample.get(B, items)  # b -> epochs with 'S' ; bi -> index of these epochs 
    i = ai+bi                         # total number of items, number of all epochs being analyzed, should be equal to items*2
    x = a + b                         # all epochs, containing 'N' epochs and 'S' epochs
    y = [0]*items + [1]*items         # structure initialitation 
    
    esppe = [preprocess(eeg, ep) for i,ep in enumerate(x)] 
    print("resampled: ", esppe[0].shape)
    """ Filter frequency bands """
    fesppe = []
    if Bands: fesppe = [band(e, bands, esppe[0].shape[1]) for e in esppe]
    elif not Bands: fesppe = esppe
    
    result = struct(x=np.array(x), y=np.array(y), i=np.array(i))
    if dict_methods[method_idx] == spectral_coherence:
        result._set(SC = connectivity_analysis(fesppe,spectral_coherence,fesppe[0].shape[1],imag))
    if dict_methods[method_idx] == PEC: 
        result._set(PEC = [PEC(ep,i+1) for i,ep in enumerate(fesppe)])
    if dict_methods[method_idx] == phaselock:
        result._set(PLV = connectivity_analysis(fesppe, phaselock))
    if dict_methods[method_idx] == phaselag:
        result._set(PLI = connectivity_analysis(fesppe, phaselag))
    if dict_methods[method_idx] == cross_correlation:
        result._set(CC = connectivity_analysis(fesppe, cross_correlation))
    if dict_methods[method_idx] == PAC:
        result._set(PAC = connectivity_analysis(fesppe, PAC, True, fesppe[0].shape[1]))
        
    if Bands: rec(result).save(preprocessed+"{}/".format(method_code[method_idx]+ext)+"{}/".format(str(bands).replace(" ",""))+"NN/"+'.'.join(['-'.join([subid,"nn"]),'prep']))
    elif not Bands: rec(result).save(preprocessed+"{}/".format(method_code[method_idx]+ext)+"NN/"+'.'.join(['-'.join([subid,"nn"]),'prep']))

    nid, nodes = list(range(len(eeg.axes.region))), list(eeg.axes.region) # nid-> identificator of each node ; nodes -> name of the electrode contact (ex: B'1-B'2)
    nxn, base, ftype, rtype = combinations(nid, 2), [], '.prep', '.res'  
    AB, pid = (A+B).lower(), subid
    pid = '-'.join([pid,AB])
    if Bands: prep_path=preprocessed+"{}/".format(method_code[method_idx]+ext)+"{}/".format(str(bands).replace(" ",""))+"NN/"+'.'.join(['-'.join([subid,"nn"]),'prep'])
    elif not Bands: prep_path=preprocessed+"{}/".format(method_code[method_idx]+ext)+"NN/"+'.'.join(['-'.join([subid,"nn"]),'prep'])

    pdata = rec.load(prep_path).data
    print('processing base combinations...', end=' ')
    for pair in nxn: base.append(enlist(pair, nodes, analyze(pdata, pair, methods = methods, dict_methods=dict_methods))) 
    print('{} done'.format(len(base)))
    base.sort(key=lambda x:x[-1], reverse=True)
    print('best hub: {}'.format(base[0][1]), end='; ')
    best, netn, sets, nets = base[0][-1], 3, base[:], []

    while netn<=max_netn:
        print('checking {} nodes'.format(netn), end='... ')
        head, tests = check_until(sets),0
        for hub in sets[:head if head>0 else 1]:
            for node in nid:
                if node not in hub[0]:
                    test = hub[0]+(node,)
                    nets.append(enlist(test, nodes, analyze(pdata, test, methods = method_idx, dict_methods=dict_methods)))
                    tests += 1
        print('{} done'.format(tests))
        nets.sort(key=lambda x:x[-1], reverse=True)
        print('best net: {}'.format(nets[0][1]), end='')
        if nets[0][-1]>=best:
            if netn<max_netn:
                best = nets[0][-1]
                sets = nets[:]
                nets = []
                print(';', end=' ')                
            netn += 1
        else: break
    selected = sorted(set([t for n in nets[:check_until(nets)] for t in n[1].split('<->')]))
    print('\nselected nodes: {}/{}.'.format(', '.join(selected), len(selected)))
    
    if Bands: rec(struct(base=base, nets=nets, nodes=selected)).save(out+"{}/".format(method_code[method_idx]+ext)+"{}/".format(str(bands).replace(" ",""))+"NN/"+'-'.join([subid,"nn"])+rtype)
    elif not Bands: rec(struct(base=base, nets=nets, nodes=selected)).save(out+"{}/".format(method_code[method_idx]+ext)+"NN/"+'-'.join([subid,"nn"])+rtype)
