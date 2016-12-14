import numpy as np
import pandas as pd

#rows are Officers, columns are Assignments
raw_pref = pd.read_csv('RawPreferences2.csv', dtype='str')

officers = range (0,160)
assignments = range(0,138)


for o in officers: #160
    pref = 1
    p = []
    d = raw_pref.iloc[o,:]
    
    for i in assignments: #min and max observed values)
	
	a = np.where(d==str(i))[0]
	#print str(i) + '\t' + str(a) +'\t'+ str(pref)
	if len(a) >0:
            p.append((a,pref))
	    pref = pref + 1
    a = np.where(d=='NaN')[0]
    if len(a) >0:
        p.append((a,pref))

    np.save('p'+str(o), p)
    #print "Officer " + str(o) + 'has preferences ' + str(p[o]) + '\n'




