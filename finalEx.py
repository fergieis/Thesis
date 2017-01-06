from assignmentFunction import *
import timeit
import sys

#print "hi"
i = 0
begin = timeit.default_timer()

sys.stdout.write('\n')
sys.stdout.flush()


#print assignmentFunction([0,0], [0,0],3)
#print assignmentFunction([1,1,1],3)
f = open('exData2.csv', 'a')
f.write("method, obj, restrictions,directeds,rejects,iter,changes, time\n")
for restrictions in range(0,5): 
    for directeds in range(0,5):
	for rejects in range(0,5):
	    for iteration in range(0,300):
		for method in range(1,4):
		    i += 1
		    t = timeit.default_timer()-begin
		    p = (i/112500.00)	
		    sys.stdout.write('{:03.3f} percent complete, estimate {:03.3f} seconds remaining         \r'.format(p*100, (t/p) - t))
		    sys.stdout.flush()
		    start_time = timeit.default_timer()
		    s = assignmentFunction([restrictions,directeds,rejects],method)
		    t = timeit.default_timer() - start_time
		    f.write(str(method)+ ','+ str(s.obj)+ ',' + str(restrictions)+','+str(directeds)+','+str(rejects)+','+str(iteration)+','+str(s.changes)+','+str(t) + '\n')
f.close()
print 'Complete!                                                           '
