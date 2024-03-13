import numpy as np
from models import NormativeEV, NormativeLogRatio


def fit(x0, args):
    # print('Running fit...')
    ntrials, s, a, r, destroy, ff1, ff2 = args
    # print(ntrials, s, a, r, destroy, ff1, ff2)
    ff_values = np.linspace(-1, 1, 10).round(1)
    try:
        _ = len(x0)
    except:
        x0 = [x0]
        
    if len(x0) == 1:
        temp = x0[0]
        m = NormativeEV(temp=temp, x=ff_values)
    else:
        perceptual_temp = x0[0]
        rl_temp = x0[1]
        m = NormativeLogRatio(perceptual_temp=perceptual_temp, rl_temp=rl_temp, x=ff_values)
    
    ll = 0

    for t in range(ntrials):
        
        ff_chosen = [ff1[t], ff2[t]][a[t]]
        m.learn_perceptual(ff_chosen, destroy[t])

        if destroy[t]:
            m.learn_value(s[t], a[t], r[t])

        ll += np.log(m.ll_of_choice(ff1[t], ff2[t], s[t], a[t]))

    return -ll


def fit2(x0, args):
    def append(s):
        with open('log.txt', 'a') as f:
            f.write(str(s)+'\n')
    
    def f1(x):
        return np.linspace(0, 1, 10000)[int(x)-1]

    def f2(x):
        return np.linspace(0, 1, 10000)[int(x)-1]

    append('Running fit...')
    append('x0[0]: '+str(x0[0]))
    append('x0[1]: '+str(x0[1]))


    ntrials, _ = args
    # print(ntrials, s, a, r, destroy, ff1, ff2)
    ll = 0

    for t in range(ntrials):
        x1, x2 = f1(x0[0]), f2(x0[1])
        
        ll += np.log(x1+x2)

    return -ll