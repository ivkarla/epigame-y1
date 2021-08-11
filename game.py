from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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
