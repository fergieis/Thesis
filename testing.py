from assignmentFunction import *
import timeit

#print "hi"
start_time = timeit.default_timer()

#print assignmentFunction([0,0], [0,0],3)
#print assignmentFunction([1,1,1],3)
f = open('log.txt', 'a')
f.write("restrictions,directeds,rejects,iter,changes\n")
for restrictions in range(0,5): 
    for directeds in range(0,5):
	for rejects in range(0,5):
	    for iteration in range(0,10):
		c = -1
		while c == -1:
		    s = assignmentFunction([restrictions,directeds,rejects],3)
		    c = s.changes
                    if c >= 0:
			print str(restrictions)+str(directeds)+str(rejects)+str(iteration)
			f.write(str(restrictions)+','+str(directeds)+','+str(rejects)+','+str(iteration)+','+str(s.changes)+'\n')
f.write('t='+str(timeit.default_timer()))
f.close()

