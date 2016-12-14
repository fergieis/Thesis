#from __future__ import division
import numpy as np
import pandas as pd
import pickle as pkl


#two helper functions to save and load python objects
def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pkl.load(f)

#Solution Class provides means of return more than one value
class Solution:
    def __init__(self, fS, rSF, c): #add msg?
        self.finalSolution = fS
	#self.computationTime = cT
	self.resultStatusFlag = rSF
	self.changes= c
	#self.resultStatusMsg = ...

    def __str__(self): #when called via print
	return str([self.finalSolution, self.computationTime, self.resultStatusFlag, self.changes])
	
    def __repr__(self): #when called interactively
	return str([self.finalSolution, self.computationTime, self.resultStatusFlag, self.changes])

def setup():
    SMAGetPrefs()
    LPGetYearGroup()
    LPGetC()

#Generates 'p.pkl' file containing Officer preference dictionary
#Only needs to be run once on a machine
#setup for use as a stack (pop/append) in reverse preference order.
#Stack contains sublists for each level of indifference (weak preference order)
def SMAGetPrefs():
    import numpy as np
    import pandas as pd
    raw_pref = pd.read_csv('RawPreferences2.csv', dtype='str')
    p_kd = {}
    p_b = {}

    KD_Off= range(74,160)
    B_Off = range(0, 74)
    KD_Ass= range(0,71)
    B_Ass = range(73,139)
    D_Ass = range(139, 162)
    KD_D_Ass = D_Ass[0:15]
    Officers = list(set().union(KD_Off,B_Off))
    Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))

    for o in Officers:
	p_kd[o] = []
	p_b[o] = []
		
	d_kd = raw_pref.iloc[o,KD_Ass]
	d_b = raw_pref.iloc[o,B_Ass]
	end =[]

	#build KD pref list- in raw, min=0, max=137
	for i in sorted(range(0,138), reverse = True):
	    #Find all indices with same preference value
    	    a = list(np.where(d_kd==str(i))[0]) + list(np.where(d_kd==i)[0])
	    if len(a) >0:
	        end.append(d_kd[a].index[0])

	#first no assignment
	p_kd[o].append(KD_D_Ass)	
	#then Nulls/missing values
	p_kd[o].append(list(map(int,d_kd[d_kd.isnull()].index)))
	#then add in reverse preference order	
	p_kd[o].append(end) 
	#p[o] is now a "stack" with iso-preference lists to which
	#Officer o is indifferent

	#build B pref list
	end =[]
	for i in sorted(range(0,138), reverse = True):
	    a = list(np.where(d_b==str(i))[0]) + list(np.where(d_b==i)[0])
	    if len(a) >0:
	        end.append(d_b[a].index[0])
	p_b[o].append(B_Ass)
	p_b[o].append(list(map(int,d_b[d_b.isnull()].index)))
	p_b[o].append(end) 

    save_obj(p_kd, 'p_kd')
    save_obj(p_b, 'p_b')

#Generates 'y.pkl' file containing yeargroup list. 
#Only needs to be run once on a machine
def LPGetYearGroup():
    d = pd.read_csv('OfficerData.csv')
    y = d['YG']
    y = y.values.tolist()
    save_obj(y, 'y')

    yg={}
    for o, year in enumerate(y):
	try:
	    yg[year].append(o)
	except (KeyError, NameError):
	    yg[year]=[o]
    save_obj(yg,'yg')

#Generates 'C.pkl' file containing yeargroup list. 
#Only needs to be run once on a machine
def LPGetC():
    KD_Off= range(74,160)
    B_Off = range(0, 74)
    KD_Ass= range(0,71)
    B_Ass = range(73,139)
    D_Ass = range(139, 162)

    Officers = list(set().union(KD_Off,B_Off))
    Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))
    C = {}
    prefs ={} 
    officer_preference = pd.read_csv('OfficerPrefs.csv')	
    for o in Officers: 
        prefs[o] = officer_preference[str(o)]
        for a in Assignments:
            if a in list(D_Ass):
		C[(o,a)] = 0
	    elif not np.isnan(prefs[o][a]):		
		C[(o,a)] = prefs[o][a]	
	    else: #mildly infeasible, no KD prefs avail
		C[(o,a)]= 999
    save_obj(C,'C')
		

def assignmentFunction(allowableChanges, methodFlag): #don't need init solution
    #from math import floor, sqrt
    #for non-integer division, convert to floats!	
    #import pandas as pd
    import numpy as np
    import gurobipy as gp
    import random as rnd
    
    
    sol = Solution([0,0], 1, 0)	


    #Because Python doesn't have select-case
    if methodFlag == 0:   #SMA"Warmstart" -- may drop
	print "zero"
    

    elif methodFlag == 1: #SMAColdstart
	

        #sets
	KD_Off= range(74,160)
	B_Off = range(0, 74)
	KD_Ass= range(0,86)
	B_Ass = range(86,140)
	D_Ass = range(140, 160)
	#Officers = list(set().union(KD_Off,B_Off))
	#Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))


	#Match KD Assignments
	KD_D_Ass = D_Ass[0:15]
	p_kd = load_obj('p_kd')
	yg = load_obj('yg')
	#need opref and apref
	
	n = len(KD_Off)
	#rankO = np.zeros(n+1).tolist() #Officer to assignment currO[assignment]
	#rankO = [-1]*(n)
	#provides ordinal ranking of how far "down" each officers list is current match
	#rankA = [-1]*(n)
	KD_OtoA = {}

	
	opref = {}
	apref = {}
	for o in KD_Off:
	    opref[o] = p_kd[o]
	    KD_OtoA[o] = -1
	
	for assignment in KD_Ass:
	    apref[assignment] = {}
	    preference = 0
	    for year in sorted(yg.keys()):
		preference += 1
		for officer in yg[year]:
			apref[assignment][officer] = preference
#	for assignment in B_Ass:
#	    apref[assignment]=[0]*(n)
#	    for officer in KD_Off:
#		apref[assignment][officer] = 2
#	    for officer in B_Off:
#		apref[assignment][officer] = 1
#	
	for assignment in KD_D_Ass:
	    apref[assignment] = [0]*(n)

	unmatched = KD_Off #listing of unmatched officers
	noKD = [] #bookkeeping for KD Officers not getting a KD Assignment

	while unmatched: #implict boolean false for empty list
	    #find first unmatched officer)
	    officer = rnd.choice(unmatched)	# = rankA[1:].index(0)+1
	    #rankA[officer] = rankA[officer] + 1
	    #don't loop, choose next best.
	    
	    #if len(opref[officer]) > 1:
	    #	possibilities = opref[officer].pop()
	    #else:
	    #	possibilities = opref[officer]	
	    
	    possibilities = opref[officer].pop()
	    print "-----\nO:" + str(officer)
	    print "p:-----"
	    print possibilities
	    
	    #if len(possibilities) >= 1:
	    #	possibility = rnd.choice(possibilities)
	    #	possibilities.remove(possibility)	    
	    #else:
	    #	possibility = possibilities[0]
	    
	    possibility = rnd.choice(possibilities)
	    possibilities.remove(possibility)
	    print possibility
	    
	    if len(possibilities) > 0: 
	    #if sublist not empty, replace on stack
		opref[officer].append(possibilities)
	    
	    #change to rank++ below
	    try:
	        possibilitypref = apref[possibility]
	    except KeyError:
		print "KEYERROR!"
		break

    	    try:
		#incumbent = KD_OtoA.index(possibility)
		incumbent = KD_OtoA.keys()[KD_OtoA.values().index(possibility)]

	    except ValueError:
		incumbent = -1
		#no incumbent	
	    if incumbent == -1:
		unmatched.remove(officer)			
		rankO[possibility] = possibilitypref[officer]
		KD_OtoA[officer] = possibility #rankA[officer] = possibility
				
				
	     #if assignment prefers officer to incumbent	
 	    elif possibilitypref[officer] > possibilitypref[incumbent]:
	    #match officer to possibility
	    #break match of possibility if needed
		KD_OtoA[incumbent] = -1
		unmatched.append(incumbent)
		KD_OtoA[officer] = possibility
		rankO[possibility] = possibilitypref[officer]
		unmatched.remove(officer)	

	    #elif possibilitypref[officer] == possibilitypref[incumbent]:
		
		
	print KD_OtoA
	sol.finalSolution = KD_OtoA
	sol.resultStatusFlag = 0
	sol.changes = 0	
	#save_obj(OtoA, 'smaA')

    elif methodFlag == 2: #LPWarmstart
	kd_m = gp.Model()	
	y = load_obj('y')
	lpA = load_obj('lpA')

        #sets
	KD_Off= range(74,160)
	B_Off = range(0, 74)
	KD_Ass= range(0,86)
	B_Ass = range(86,140)
	D_Ass = range(140, 160)
	Officers = list(set().union(KD_Off,B_Off))
	Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))
	
        #handlemods

	#allowableChanges[0]: Assignment Restrictions	
	restrictions = rnd.sample(Officers, allowableChanges[0])
	A_r = {}
	for restriction in restrictions:
	    A_r[restriction] = rnd.sample(Assignments, int(rnd.uniform(.05,.1) * len(Assignments)))

	#allowableChanges[1]: Directed Assignment
	A_d = {}
	availO = set(Officers)
	availO -= set(restrictions)
        directeds = rnd.sample(list(availO), allowableChanges[1])
	availA = set(Assignments)
	for directed in directeds:
	    availA -= set(A_d.values())
	    A_d[directed] = rnd.sample(list(availA), 1)
	    A_d[directed]=A_d[directed][0]

	#allowableChanges[2]: Rejected Match
	availO -= set(directeds)
	rejects = []
	while len(rejects) < allowableChanges[2]:
	    reject = rnd.sample(list(availO), 1)[0]
	    if int(lpA[reject]) != 999:
		rejects.append(reject)
		availO.remove(reject)

	#allowableChanges[3]: Unforecast Assignment
	#Need to remove all references, these are just the dummy assignments
	#mildly embarassing... but reference in write up

	
	#Phase I, slot KD Offs/objective  f_1
	x={}
	for o in KD_Off:
	    for a in KD_Ass:
		x[(o,a)] = kd_m.addVar(vtype=gp.GRB.BINARY, 
			   obj = y[o],
                           name="O{0:03d}".format(o) + 
			   "A{0:03d}".format(a))
		if lpA[o] == a:
			x[(o,a)].Start = 1
		else:
			x[(o,a)].Start = 0

	kd_m.ModelSense = gp.GRB.MINIMIZE #MINIMIZE

        for a in KD_Ass:
            kd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in KD_Off),gp.GRB.EQUAL,1)	
	for o in KD_Off:
	    if o in list(restrictions):
		if not list(set(A_r[o])&set(KD_Ass)):
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in list(set(A_r[o])&set(KD_Ass))), gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)  
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,0)
	    elif o in list(directeds):
		if A_d[o] in list(KD_Ass):
	            kd_m.addConstr(x[o,A_d[o]], gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,0)
	    elif o in list(rejects):
		if lpA[o] in list(KD_Ass):
		    kd_m.addConstr(x[o,int(lpA[o])], gp.GRB.EQUAL,0)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)
	    else:
	        kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)

	kd_m.update()
	kd_m.setParam('OutputFlag', False)
	#kd_m.write('lp.mps')
	kd_m.optimize()
	#print kd_m.Status
	try:
	    y_star = kd_m.objVal
	except:
	    return Solution(-1* np.ones(160), -1, -1)
	#Continue with phase2, slot everyone- obj f_2
        C = load_obj('C')
	x = {} #reallocating memory
	bd_m = gp.Model()

        for o in Officers: 
	    for a in Assignments:
		x[(o,a)] = bd_m.addVar(vtype=gp.GRB.BINARY, obj=C[(o,a)],name="O{0:03d}".format(o) + "A{0:03d}".format(a))
		
        bd_m.ModelSense = gp.GRB.MINIMIZE #-1
	for a in Assignments:
            bd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in Officers),gp.GRB.EQUAL,1)	
	for o in Officers:
	    if o in list(restrictions):
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in A_r[o]), gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(directeds):
	        bd_m.addConstr(x[o,A_d[o]], gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(rejects):
		bd_m.addConstr(x[o,lpA[o]], gp.GRB.EQUAL,0)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    else:
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	
	bd_m.addConstr(gp.quicksum(gp.quicksum(y[o]*x[(o,a)] for a in KD_Ass) for o in Officers), gp.GRB.LESS_EQUAL, 1.0015* y_star) 
	bd_m.setParam('OutputFlag', False)
	bd_m.update()
	bd_m.optimize()
	#print bd_m.Status
	#unassigned =[]
	sol.finalSolution = np.zeros(160)
        try:
	    for v in bd_m.getVars():
	        if v.x >0:
	            if int(v.varName[-3:])>=140:
	                #unassigned.append(int(v.varName[1:4]))
		        sol.finalSolution[int(v.varName[1:4])] = 999
		    else:		
		        sol.finalSolution[int(v.varName[1:4])] = int(v.varName[-3:])
	    sol.finalSolution = sol.finalSolution.astype(int)
	    sol.resultStatusFlag = 0 #Good Execution
	    #save_obj(sol.finalSolution, 'lpA') #saved locally first time and referenced later
	    #lpA = load_obj('lpA')
	    sol.changes = sum(i != j for i, j in zip(sol.finalSolution, lpA))
        except:
	    sol = Solution(-1* np.ones(160), -1, -1) 





    elif methodFlag == 3: #LPColdstart
	kd_m = gp.Model()	
	y = load_obj('y')
	lpA = load_obj('lpA')

        #sets
	KD_Off= range(74,160)
	B_Off = range(0, 74)
	KD_Ass= range(0,86)
	B_Ass = range(86,140)
	D_Ass = range(140, 160)
	Officers = list(set().union(KD_Off,B_Off))
	Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))
	
        #handlemods

	#allowableChanges[0]: Assignment Restrictions	
	restrictions = rnd.sample(Officers, allowableChanges[0])
	A_r = {}
	for restriction in restrictions:
	    A_r[restriction] = rnd.sample(Assignments, int(rnd.uniform(.05,.1) * len(Assignments)))

	#allowableChanges[1]: Directed Assignment
	A_d = {}
	availO = set(Officers)
	availO -= set(restrictions)
        directeds = rnd.sample(list(availO), allowableChanges[1])
	availA = set(Assignments)
	for directed in directeds:
	    availA -= set(A_d.values())
	    A_d[directed] = rnd.sample(list(availA), 1)
	    A_d[directed]=A_d[directed][0]

	#allowableChanges[2]: Rejected Match
	availO -= set(directeds)
	rejects = []
	while len(rejects) < allowableChanges[2]:
	    reject = rnd.sample(list(availO), 1)[0]
	    if int(lpA[reject]) != 999:
		rejects.append(reject)
		availO.remove(reject)

	#allowableChanges[3]: Unforecast Assignment
	#Need to remove all references, these are just the dummy assignments
	#mildly embarassing... but reference in write up

	
	#Phase I, slot KD Offs/objective  f_1
	x={}
	for o in KD_Off:
	    for a in KD_Ass:
		x[(o,a)] = kd_m.addVar(vtype=gp.GRB.BINARY, 
			   obj = y[o],
                           name="O{0:03d}".format(o) + 
			   "A{0:03d}".format(a))
	kd_m.ModelSense = gp.GRB.MINIMIZE #MINIMIZE

        for a in KD_Ass:
            kd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in KD_Off),gp.GRB.EQUAL,1)	
	for o in KD_Off:
	    if o in list(restrictions):
		if not list(set(A_r[o])&set(KD_Ass)):
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in list(set(A_r[o])&set(KD_Ass))), gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)  
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,0)
	    elif o in list(directeds):
		if A_d[o] in list(KD_Ass):
	            kd_m.addConstr(x[o,A_d[o]], gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,0)
	    elif o in list(rejects):
		if lpA[o] in list(KD_Ass):
		    kd_m.addConstr(x[o,int(lpA[o])], gp.GRB.EQUAL,0)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)
	    else:
	        kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Ass),gp.GRB.EQUAL,1)

	kd_m.update()
	kd_m.setParam('OutputFlag', False)
	#kd_m.write('lp.mps')
	kd_m.optimize()
	#print kd_m.Status
	try:
	    y_star = kd_m.objVal
	except:
	    return Solution(-1* np.ones(160), -1, -1)
	#Continue with phase2, slot everyone- obj f_2
        C = load_obj('C')
	x = {} #reallocating memory
	bd_m = gp.Model()

        for o in Officers: 
	    for a in Assignments:
		x[(o,a)] = bd_m.addVar(vtype=gp.GRB.BINARY, obj=C[(o,a)],name="O{0:03d}".format(o) + "A{0:03d}".format(a))
		
        bd_m.ModelSense = gp.GRB.MINIMIZE #-1
	for a in Assignments:
            bd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in Officers),gp.GRB.EQUAL,1)	
	for o in Officers:
	    if o in list(restrictions):
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in A_r[o]), gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(directeds):
	        bd_m.addConstr(x[o,A_d[o]], gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(rejects):
		bd_m.addConstr(x[o,lpA[o]], gp.GRB.EQUAL,0)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    else:
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	
	bd_m.addConstr(gp.quicksum(gp.quicksum(y[o]*x[(o,a)] for a in KD_Ass) for o in Officers), gp.GRB.LESS_EQUAL, 1.0015* y_star) 
	bd_m.setParam('OutputFlag', False)
	bd_m.update()
	bd_m.optimize()
	#print bd_m.Status
	#unassigned =[]
	sol.finalSolution = np.zeros(160)
        try:
	    for v in bd_m.getVars():
	        if v.x >0:
	            if int(v.varName[-3:])>=140:
	                #unassigned.append(int(v.varName[1:4]))
		        sol.finalSolution[int(v.varName[1:4])] = 999
		    else:		
		        sol.finalSolution[int(v.varName[1:4])] = int(v.varName[-3:])
	    sol.finalSolution = sol.finalSolution.astype(int)
	    sol.resultStatusFlag = 0 #Good Execution
	    #save_obj(sol.finalSolution, 'lpA') #saved locally first time and referenced later
	    #lpA = load_obj('lpA')
	    sol.changes = sum(i != j for i, j in zip(sol.finalSolution, lpA))
        except:
	    sol = Solution(-1* np.ones(160), -1, -1) 

    else: #Error
	print "Parameter Error: Invalid methodFlag"        
	sol = Solution(-1* np.ones(160), -1, -1)	
	
    return sol





#If called from a command shell, call self
#if __name__ == "__main__":
#    import sys
#    assignmentFunction(sys.argv[1])
