from collections import deque
import numpy as np
 
class SimpleMovingAverage():
    def __init__(self, period):
        assert period == int(period) and period > 0, "Period must be an integer >0"
        self.period = period
        self.stream = deque()
 
    def __call__(self, n):
        stream = self.stream
        stream.append(n)    # appends on the right
        streamlength = len(stream)
        if streamlength > self.period:
            stream.popleft()
            streamlength -= 1
        if streamlength == 0:
            average = 0
        else:
            average = sum( stream ) / float(streamlength)
 
        return average

class Lagger():
    def __init__(self, lag):
        self.lag = lag
        self.stream = deque()
 
    def __call__(self, n):
        stream = self.stream
        stream.append(n)    # appends on the right
        streamlength = len(stream)
        if streamlength <= self.lag+1:
            pass
        else:
            stream.popleft()
        
        return stream[0]
    
class CountWhile():
    def __init__(self, criteria):
        self.stream = list()
        self.criteria = criteria
 
    def __call__(self, n):
        stream = self.stream
        streamlength = 0
        if eval('n'+self.criteria):
            stream.append(n)    # appends on the right
            streamlength = len(stream)
        else:
            self.stream = list()
        return streamlength

def tick(streamers, newdata):
    return([c(newdata[num]) for num, c in enumerate(streamers)])

def init_lag(nunits, name, var, lag):
    d = {'name': name, 
         'var': var,
         'streamers': [Lagger(lag) for streamer in range(nunits)]}
    return(d)

def init_count_while(nunits, name, var, cw):
    d = {'name': name, 
         'var': var,
         'streamers': [CountWhile(cw) for streamer in range(nunits)]}
    return(d)

def init_moving_average(nunits, name, var, ma):
    d = {'name': name, 
         'var': var, 
         'streamers': [SimpleMovingAverage(ma) for streamer in range(nunits)]}
    return(d)

def init_order(nunits, tsvar):
    if 'lag' in tsvar.keys():
         d = init_lag(nunits, **tsvar)
    elif 'cw' in tsvar.keys():
        d = init_count_while(nunits, **tsvar)
    elif 'ma' in tsvar.keys():
        d = init_moving_average(nunits, **tsvar)
    else:
        raise TypeError
    return(d)

def update_df(df, t, stream):
    df.loc[t, stream['name']] = tick(stream['streamers'], df.loc[t, stream['var']].values)
