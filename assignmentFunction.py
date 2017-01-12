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
	self.obj = 0.0
	#self.resultStatusMsg = ...

    def __str__(self): #when called via print
	return str([self.finalSolution, self.obj, self.resultStatusFlag, self.changes])
	
    def __repr__(self): #when called interactively
	return str([self.finalSolution, self.obj, self.resultStatusFlag, self.changes])


def setup():
    LPGetYearGroup()
    SMAGetPrefs()
    LPGetC()

#Generates 'p.pkl' file containing Officer preference dictionary
#Only needs to be run once on a machine
#setup for use as a stack (pop/append) in reverse preference order.
#Stack contains sublists for each level of indifference (weak preference order)
def SMAGetPrefs():


    import numpy as np
    import pandas as pd
    raw_pref = pd.read_csv('RawPreferences2.csv')#, dtype='str')
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

    for o in KD_Off:
	d_kd = raw_pref.iloc[o,KD_Ass]
	p_kd[o] = KD_D_Ass + list(map(int,d_kd.sort_values(ascending=False).index))
    for o in Officers:
	d_b = raw_pref.iloc[o,B_Ass]
	d_b = list(map(int,d_b.sort_values(ascending=False).index))
	p_b[o] = D_Ass + d_b
	
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

#Generates 'C.pkl' file containing cost coefficient list. 
#Only needs to be run once on a machine
def LPGetC():
    KD_Off= range(74,160)
    B_Off = range(0, 74)
    KD_Ass= range(0,71)
    B_Ass = range(73,139)
    D_Ass = range(139, 162)

    y = load_obj('y')
    years = []
    for i in y:
	years.append(i-min(y))


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
		if o in KD_Off and a in KD_Ass:
		    penalty = 0
		elif o in KD_Off and a in B_Ass:
		    penalty = 160 * (10 - years[o])
		elif o in B_Off and a in KD_Ass:
		    penalty = 160 * years[o]
	        elif o in B_Off and a in B_Ass:		
		    penalty = 0
		else:
		    pass
		C[(o,a)] = prefs[o][a] + penalty
	    else: #mildly infeasible, no KD prefs avail
		C[(o,a)]= 999
    save_obj(C,'C')


#SMA  SMA  SMASMA  SMA  SMASMA  SMA  SMASMA  SMA  SMASMA  SMA  SMA
# SMA  SMA  SMASMA  SMA  SMASMA  SMA  SMASMA  SMA  SMASMA  SMA  SMA

def SMA(allowableChanges):
    import random as rnd
    sol = Solution([0,0], 1, 0)	
    #try:
    if 1:
        #sets
        KD_Off= range(74,160)
        KD_Ass= range(0,71)
	B_Off = range(0, 74)
	B_Ass = range(73,139)
	D_Ass = range(139, 162)	
	Officers = list(set().union(KD_Off,B_Off))
	Assignments = list(set().union(KD_Ass, B_Ass))

	#Match KD Assignments
	KD_D_Ass = D_Ass[0:15]
	p_kd = load_obj('p_kd')
	p_b = load_obj('p_b')
	yg = load_obj('yg')
	y = load_obj('y')
	smaA = load_obj('smaA')
	KD_OtoA = {}
	BOtoA = {}

	#allowableChanges[0]: Assignment Restrictions	
	availO = set(Officers)	
	restrictions = rnd.sample(list(availO), allowableChanges[0])
	for restriction in restrictions:
	    res_list = rnd.sample(Assignments, int(rnd.uniform(.05,.1) * len(Assignments)))
	    #print res_list
	    for ass in res_list:
		if ass in KD_Ass and restriction in KD_Off:
		    p_kd[restriction].remove(ass)
		    p_kd[restriction] = [ass] + p_kd[restriction]
		elif ass in B_Ass:
		    p_b[restriction].remove(ass)
		    p_b[restriction] = [ass] + p_b[restriction]
		else:
                    pass	        



	#allowableChanges[1]: Directed Assignment
	taken = []
	x = {}
	availO -= set(restrictions)
	availA = set().union(KD_Ass, B_Ass, D_Ass)
        directeds = rnd.sample(list(availO), allowableChanges[1])
	for directed in directeds:
	    #if directed in KD_Off:
		#KD_Off.remove(directed)
	    #else:
		#B_Off.remove(directed)
	    x[directed]=rnd.sample(list(availA), 1)[0]
	    if directed in KD_Off:
		x[directed]=rnd.sample(list(set(availA)-set(B_Ass)), 1)[0]
		KD_OtoA[directed] = x[directed]	
	    else:
		x[directed]=rnd.sample(list(set(availA)-set(KD_Ass)), 1)[0]
            	BOtoA[directed] = x[directed]
	    taken += [x[directed]]
	    availA -= set(taken)


	#allowableChanges[2]: Rejected Match
	availO -= set(directeds)
	rejects = []
	while len(rejects) < allowableChanges[2]:
	    reject = rnd.sample(list(availO), 1)[0]
	    if int(smaA[reject]) != 999:
		rejects.append(reject)
		availO.remove(reject)
		if reject in KD_Off and smaA[reject] in KD_Ass:
		    p_kd[reject].remove(smaA[reject])
		    p_kd[reject] = [smaA[reject]] + p_kd[reject]
		else: #if reject in KD_Off and smaA[reject] in B_Ass:		
		    p_b[reject].remove(smaA[reject])
		    p_b[reject] = [smaA[reject]] + p_b[reject]


			
		
	del availO
	del availA

	#Starting KD matching 
	opref = {}
	apref = {}
	unmatched = list(set(KD_Off)-set(directeds)) #listing of unmatched officers
	noKD = [] #bookkeeping for KD Officers not getting a KD Assignment
	#kdNotTaken = list(set().union(set(KD_Ass),set(KD_D_Ass) )- set(taken))
	#for o in KD_Off:

	
	for assignment in KD_Ass:
	    apref[assignment] = {}
	    preference = 0
	    for year in sorted(yg.keys()):
		preference += 1
		for officer in yg[year]:
		    apref[assignment][officer] = preference
	for o in directeds:
	    if x[o] in KD_Ass:
		apref[x[o]][o] = -9999
		
	for assignment in KD_D_Ass:
	    apref[assignment] = {}
	    for officer in KD_Off:
		opref[officer] = p_kd[officer]
	        KD_OtoA[officer] = -1
		apref[assignment][officer] = -y[officer] #tour equity? yg?
	
	while unmatched:
	    #officer = rnd.choice(unmatched)
	    officer = unmatched.pop()
	    #try:
	    if len(opref[officer]) >1:
		possibility = opref[officer].pop()
	    else:
		intersect = (set(KD_Ass).union(set(KD_D_Ass)))-set(KD_OtoA.values())
		possibility = rnd.choice(list(intersect))
	    	#print "intersect meth:" + str(intersect) + '\t'+str(possibility)
		#print directeds	
		#print x	
		#print possibility not in taken
		#print officer not in rejects or possibility != smaA[officer]
		try:
		    incumbent = KD_OtoA.keys()[KD_OtoA.values().index(possibility)]
		except ValueError:
		    incumbent = -1	
		#print "i:" + str(incumbent)
		break
	    #except IndexError:
#		print "IndexError"		
#		if unmatched == []:
#		    
#		    KD_OtoA[officer] = list(intersect)[0]
#		    break
#		else:
 #		    pass


		

#		if len(kdNotTaken) >0:
#		    possibility = rnd.choice(kdNotTaken)
#		else:
#		    
#		    intersect = set(KD_Ass).union(set(KD_D_Ass))-set(KD_OtoA.values())-set([-1])
#		    print 'unmatched:'+str(unmatched)
#		    print intersect
#		    if len(intersect)>0:
#		        KD_OtoA[officer]= list(intersect)[0]
#		        taken += [KD_OtoA[officer]]  
#		    print "Fatal error?"   
#		    print str(officer) + str(KD_OtoA[officer])
#		    print "KD Avail:" + str(kdNotTaken)
#		    override = True
			

	    #condition1 = possibility not in taken
	    #condition2 = 
	    if officer not in rejects or possibility != smaA[officer]:
		possibilitypref = apref[possibility]
      	        try:
		    incumbent = KD_OtoA.keys()[KD_OtoA.values().index(possibility)]
		except ValueError:
		    incumbent = -1
			

		#no incumbent	
		if incumbent == -1:
		    #unmatched.remove(officer)			
		    KD_OtoA[officer] = possibility
		    #kdNotTaken.remove(possibility)# += [possibility]
		    if possibility in KD_D_Ass:
		        noKD.append(officer)
				
		#if assignment prefers officer to incumbent	
	 	elif possibilitypref[officer] < possibilitypref[incumbent]:
		    KD_OtoA[incumbent] = -1
		    if possibility in KD_D_Ass:
			noKD.remove(incumbent)
			noKD.append(officer)
		    unmatched.append(incumbent)
		    KD_OtoA[officer] = possibility
		    #unmatched.remove(officer)	
		else:
		    unmatched = [officer] + unmatched
	    else:
		unmatched = [officer] + unmatched

#-----------------------------------------
	del p_kd	#other garbage cleanup
	del yg


	#Setup for Broad. Matching
	opref = {}
	apref = {}

	btaken = list(taken)
	p_b = load_obj('p_b')
	unmatched = list(set(noKD + B_Off)-set(directeds))
	for o in unmatched:
	    opref[o] = p_b[o]
	    BOtoA[o] = -1
	
	for assignment in B_Ass:
	    apref[assignment]={}
	    for officer in unmatched:
		if officer in B_Off:
		    apref[assignment][officer] = 1
		else:
		   apref[assignment][officer] = 2   
	
   	for assignment in D_Ass:
	    apref[assignment] = {}
	    for officer in unmatched:
		apref[assignment][officer] = 0
	
	for o in directeds:
	    if x[o] in B_Off:
		apref[x[o]]= apref[160]
		apref[x[o]][o] = -9999

	while unmatched: #implict boolean false for empty list
	    #officer = rnd.choice(unmatched)
	    officer = unmatched.pop()
	    try:
		possibility = opref[officer].pop()
	    except IndexError:
		#empty list
		availO = set(B_Ass+D_Ass)-set(btaken)
		if len(availO) >0:
		    possibility = rnd.choice(list(availO))
		else:
		    pass		
		    #print "Fatal Error2?"
		    #print unmatched
		    #print sorted(list(availO))
		    #print sorted(taken)


	    condition1 = possibility not in taken
	    condition2 = officer not in rejects or possibility != smaA[officer]
	    if condition1 and condition2:
	        possibilitypref = apref[possibility]
    	        try:
		    incumbent = BOtoA.keys()[BOtoA.values().index(possibility)]

	    
	        except ValueError:
		    incumbent = -1
	        if incumbent == -1:
		    #unmatched.remove(officer)	
		    btaken += [possibility]		
		    BOtoA[officer] = possibility
				
	        #if assignment prefers officer to incumbent	
 	        elif possibilitypref[officer] < possibilitypref[incumbent]:
		    BOtoA[incumbent] = -1
		    unmatched.append(incumbent)
		    BOtoA[officer] = possibility
		    #unmatched.remove(officer)	

		else:
		    unmatched.append(officer)
	    else:
		unmatched.append(officer)


#+++++++++++++++++++++++++++++++++++++++++++++++++
	for o in KD_Off:
	    if o in noKD:
		x[o] = BOtoA[o]
	    else:
		x[o] = KD_OtoA[o]
	for o in B_Off:
	    x[o] = BOtoA[o]
	    #print "O:"+str(o)+" A:"+str(KD_OtoA[o])
	sol.finalSolution = []
	Officers = list(set().union(KD_Off,B_Off))
	for i in Officers:
	    if x[i] in D_Ass or x[i] == -1:
	   	sol.finalSolution.append(int(999))
	    else:	
	        sol.finalSolution.append(x[i])

	sol.resultStatusFlag = 0
	#save_obj(sol.finalSolution, 'smaA') #saved first time, used later
		
	sol.changes = [sum(i != j for i, j in zip(sol.finalSolution, smaA))][0]

    #except IndexError:
    #	print "Index Error"        
    #	sol = Solution(-1* np.ones(160), -1, -1)	
	
    return sol

#AFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAF
#AFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAF

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
	sol = SMA(allowableChanges)

    elif methodFlag == 2: #LPWarmstart
	kd_m = gp.Model()	
	y = load_obj('y')
	lpA = load_obj('lpA')

        #sets
        KD_Off= range(74,160)
        B_Off = range(0, 74)
        KD_Ass= range(0,71)
        B_Ass = range(73,139)
        D_Ass = range(139, 162)
	D_KD_Ass = range(139,154)
	
	Officers = list(set().union(KD_Off,B_Off))
	KD_Assignments = list(set().union(KD_Ass,D_KD_Ass))	
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
	    for a in KD_Assignments:
		x[(o,a)] = kd_m.addVar(vtype=gp.GRB.BINARY, 
			   obj = y[o],
                           name="O{0:03d}".format(o) + 
			   "A{0:03d}".format(a))
		#if lpA[o] == a:
		#	x[(o,a)].Start = 1
		#else:
		#	x[(o,a)].Start = 0

	kd_m.ModelSense = gp.GRB.MINIMIZE #MINIMIZE

        for a in KD_Assignments:
            kd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in KD_Off),gp.GRB.EQUAL,1)	
	for o in KD_Off:
	    if o in list(restrictions):
		if not list(set(A_r[o])&set(KD_Assignments)):
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in list(set(A_r[o])&set(KD_Ass))), gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)  
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,0)
	    elif o in list(directeds):
		if A_d[o] in list(KD_Ass):
	            kd_m.addConstr(x[(o,A_d[o])], gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,0)
	    elif o in list(rejects):
		if int(lpA[o]) in list(KD_Ass):
		    kd_m.addConstr(x[(o,int(lpA[o]))], gp.GRB.EQUAL,0)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
	    else:
	        kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)

	kd_m.update()
	kd_m.setParam('OutputFlag', False)
	#kd_m.write('lp.mps')
	kd_m.optimize()
	try:
	    y_star = kd_m.objVal
	except:
	    #print "Error in y_star"    
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
	        bd_m.addConstr(x[(o,A_d[o])], gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(rejects):
		bd_m.addConstr(x[(o,int(lpA[o]))], gp.GRB.EQUAL,0)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    else:
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	
#	bd_m.addConstr(gp.quicksum(gp.quicksum(y[o]*x[(o,a)] for a in KD_Ass) for o in Officers), gp.GRB.LESS_EQUAL, 1.0015* y_star) 
	bd_m.addConstr(gp.quicksum(gp.quicksum(y[o]*x[(o,a)] for a in KD_Ass) for o in Officers), gp.GRB.LESS_EQUAL, y_star)
	bd_m.setParam('OutputFlag', False)
	bd_m.update()
	bd_m.optimize()
	#print bd_m.Status
	#unassigned =[]
	sol.finalSolution = np.zeros(160)
        #try:
	for v in bd_m.getVars():
	        if v.x >0:
	            if int(v.varName[-3:])>=139 or int(v.varName[-3:]) in [71,72]:
	                #unassigned.append(int(v.varName[1:4]))
		        sol.finalSolution[int(v.varName[1:4])] = 999
		    else:		
		        sol.finalSolution[int(v.varName[1:4])] = int(v.varName[-3:])
	sol.finalSolution = sol.finalSolution.astype(int)
	sol.resultStatusFlag = 0 #Good Execution
	#save_obj(sol.finalSolution, 'lpA') #saved locally first time and referenced later
	    #lpA = load_obj('lpA')
	sol.changes = sum(i != j for i, j in zip(sol.finalSolution, lpA))
	sol.obj = bd_m.objVal

        #except:
	#    sol = Solution(-1* np.ones(160), -1, -1) 





    elif methodFlag == 3: #LPColdstart
	kd_m = gp.Model()	
	y = load_obj('y')
	lpA = load_obj('lpA')

        #sets
        KD_Off= range(74,160)
        B_Off = range(0, 74)
        KD_Ass= range(0,71)
        B_Ass = range(73,139)
        D_Ass = range(139, 162)
	D_KD_Ass = range(139,154)
	Officers = list(set().union(KD_Off,B_Off))
	KD_Assignments = list(set().union(KD_Ass,D_KD_Ass))	
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
	    for a in KD_Assignments:
		x[(o,a)] = kd_m.addVar(vtype=gp.GRB.BINARY, 
			   obj = y[o],
                           name="O{0:03d}".format(o) + 
			   "A{0:03d}".format(a))
	kd_m.ModelSense = gp.GRB.MINIMIZE #MINIMIZE

        for a in KD_Assignments:
            kd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in KD_Off),gp.GRB.EQUAL,1)	
	for o in KD_Off:
	    if o in list(restrictions):
		if not list(set(A_r[o])&set(KD_Assignments)):
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in list(set(A_r[o])&set(KD_Ass))), gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)  
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,0)
	    elif o in list(directeds):
		if A_d[o] in list(KD_Assignments):
	            kd_m.addConstr(x[(o,A_d[o])], gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,0)
	    elif o in list(rejects):
		if int(lpA[o]) in list(KD_Ass):
		    kd_m.addConstr(x[(o,int(lpA[o]))], gp.GRB.EQUAL,0)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
	    else:
	        kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)

	kd_m.update()
	kd_m.setParam('OutputFlag', False)
	#kd_m.write('lp.mps')
	kd_m.optimize()
	#print kd_m.Status
	try:
	    y_star = kd_m.objVal
	except:
	    #print "Error in y_star"
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
	        bd_m.addConstr(x[(o,A_d[o])], gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(rejects):
		bd_m.addConstr(x[(o,int(lpA[o]))], gp.GRB.EQUAL,0)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    else:
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	
	bd_m.addConstr(gp.quicksum(gp.quicksum(y[o]*x[(o,a)] for a in KD_Ass) for o in Officers), gp.GRB.LESS_EQUAL, y_star) 
	bd_m.setParam('OutputFlag', False)
	bd_m.update()
	bd_m.optimize()
	#print bd_m.Status
	#unassigned =[]
	sol.finalSolution = np.zeros(160)
	
        try:
	    for v in bd_m.getVars():
	        if v.x >0:
	            if int(v.varName[-3:])>=139 or int(v.varName[-3:]) in [71,72]:
	                #unassigned.append(int(v.varName[1:4]))
		        sol.finalSolution[int(v.varName[1:4])] = 999
		    else:		
		        sol.finalSolution[int(v.varName[1:4])] = int(v.varName[-3:])
	    sol.finalSolution = sol.finalSolution.astype(int)
	    sol.resultStatusFlag = 0 #Good Execution
	    #save_obj(sol.finalSolution, 'lpA') #saved locally first time and referenced later
	    sol.changes = sum(i != j for i, j in zip(sol.finalSolution, lpA))
	    sol.obj = bd_m.objVal
        except:
	    sol = Solution(-1* np.ones(160), -1, -1) 

    else: #Error
	print "Parameter Error: Invalid methodFlag"        
	sol = Solution(-1* np.ones(160), -1, -1)	

    if methodFlag == 1:
	    C = load_obj('C')
	    sol.obj = 0
	    for i,j in enumerate(sol.finalSolution):
		if j == 999:
		    pass
		else:
		    try:	
			sol.obj = sol.obj + C[(i,j)]
		    except KeyError: #infeasible solution
			print (i,j)
			sol.obj = -1


    	
    return sol





#If called from a command shell, call self
#if __name__ == "__main__":
#    import sys
#    assignmentFunction(sys.argv[1])
