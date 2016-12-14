from __future__ import division
#from math import floor, sqrt

import pandas as pd
import numpy as np

from gurobipy import *


#KDO = []
#for i in range(74,160):
#	KDO.append(str(i))


KD_officer_preference = pd.read_csv('KDPrefs.csv')
#KD_officer_preference = KD_officer_preference.T
KD_officer_preference = KD_officer_preference[-86:]




KD_prefs = {}
C={}

for officer in range(74,160): #KD_officer_preference:	
	KD_prefs[officer] = KD_officer_preference[str(officer)]
	for assignment in range(0,86):
		#print str(officer)+ "\t"+ str(assignment)
		if assignment < 71:
			C[(officer,assignment)] = KD_prefs[officer][assignment]	
		else:	
			C[(officer,assignment)] = 72


np.savetxt('kdC.txt',KD_officer_preference, delimiter=',', header='C values for KD assignment problem')

