from assignmentFunction import *
import timeit

#print "hi"
start_time = timeit.default_timer()

#print assignmentFunction([0,0], [0,0],3)
#print assignmentFunction([1,1,1],3)
#f = open('log.txt', 'a')
print("restrictions\tdirecteds\trejects\titer\tchanges\n")
for restrictions in range(0,2): 
    for directeds in range(0,2):
	for rejects in range(0,2):
	    for iteration in range(0,2):
		for method in range(1,4):
		    s = assignmentFunction([restrictions,directeds,rejects],method)

		    print str(restrictions)+'\t'+str(directeds)+'\t'+str(rejects)+'\t'+str(iteration)+'\n'
#		f.write(str(restrictions)+'\t'+str(directeds)+','+str(rejects)+','+str(iteration)+','+str(s.changes)+'\n')
#.write('t='+str(timeit.default_timer()))
#.close()

