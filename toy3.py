
#~/Desktop/

#import sys
import pandas as pd, numpy as np
import random as rnd

n = 50

opref = {}
apref = {}


#rankO = np.zeros(n+1).tolist() #Officer to assignment currO[assignment]
rankO = [-1]*(n)
#provides ordinal ranking of how far "down" each officers list is current match
rankA = [-1]*(n)
unmatched = range(0, n)
OtoA = [-1]*(n)

#rankA = np.zeros(6).tolist() #Assignment to officer rankA[officer]
#prevA = np.zeros(50).tolist() #pointer to last match on 

for officer in xrange(0,n):
	#builds officer preference list for assignments 1-50
	#returns list, [0, "1st loc rank ", "2nd loc rank",..]
	#opref[0]=0
	opref[officer] = np.random.permutation(range(0,n)).tolist()
for assignment in xrange(0,n):
	#builds Army preferences of officers to each assignment
	#returns list [0, "1st O Rank", "2nd O Rank", ...]	
	apref[assignment] = np.random.permutation(range(0,n)).tolist()


#while there is an unassigned officer
while unmatched: #implict boolean false for empty list
 #any(assign==0 for assign in rankA[1:]):

	#find first unmatched officer)
	officer = rnd.choice(unmatched)	# = rankA[1:].index(0)+1
	
	rankA[officer] = rankA[officer] + 1
	#don't loop, choose next best.
	possibilities = opref[officer]	
	possibility = possibilities[rankA[officer]] #rnd.choice(possibilities)
		
	#change to rank++ below	

	possibilitypref = apref[possibility]	
	try:
		incumbent = OtoA.index(possibility)
	except ValueError:
		incumbent = -1

	#no incumbent	
	if incumbent == -1:
		unmatched.remove(officer)			
		rankO[possibility] = possibilitypref.index(officer)
		OtoA[officer] = possibility #rankA[officer] = possibility
		
		
	#if assignment prefers officer to incumbent			
	elif possibilitypref[officer] > possibilitypref[incumbent]:
		#match officer to possibility
		#break match of possibility if needed
		OtoA[incumbent] = -1
		unmatched.append(incumbent)
			
		OtoA[officer] = possibility
		rankO[possibility] = possibilitypref.index(officer)
		unmatched.remove(officer)			
	
print unmatched
print "\t" + str(range(0,n))
print "corr\t" + str(np.corrcoef(rankA, rankO))
print " rankA\t" + str(rankA)
print "rankO\t" + str(rankO)
print "OtoA\t" + str(OtoA)

