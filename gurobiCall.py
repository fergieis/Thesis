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
#Reference as:
#KD_prefs[74][70] #71
m = Model("BasicAssignmentLP")
x={}

for officer in range(74,160):
	for assignment in range(0,86):
		x[(officer,assignment)] = m.addVar(vtype=GRB.BINARY, name = "O{0:03d}".format(officer) + "A{0:03d}".format(assignment))

		
m.setObjective(quicksum(x[p]*C[p] for p,i in C.items()),GRB.MINIMIZE) 
		
#all officers assigned
for officer in range(74,160):
	m.addConstr(quicksum(x[(officer,assignment)] for assignment in range(0,86)),GRB.EQUAL,1)

#all assignments used
for assignment in range(0,86):
	m.addConstr(quicksum(x[(officer,assignment)] for officer in range(74,160)),GRB.EQUAL,1)

m.update()
m.write("out.lp")
m.optimize()

#sol=""
unassigned = []
sol={}
for v in m.getVars():
	if v.x >0:
		if int(v.varName[-2:])>=71:
			#print v.varName + ": Unassigned"
			unassigned.append(int(v.varName[1:4]))
		else:		
			#sol += v.varName + "=" + str(v.x)+"\n" 
			sol[int(v.varName[1:4])] = v.varName[-2:]
#print sol
#print unassigned
print sol
