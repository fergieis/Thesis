import assignmentFunction as AF

#AF.setup()

print "Length1: "+ str(len(AF.assignmentFunction([0,0,0],1).finalSolution))
print "Length2: "+ str(len(AF.assignmentFunction([0,0,1],1).finalSolution))
print "Length3: "+ str(len(AF.assignmentFunction([0,1,0],1).finalSolution))
print "Length4: "+ str(len(AF.assignmentFunction([1,0,0],1).finalSolution))
print "Length5: "+ str(len(AF.assignmentFunction([1,0,1],1).finalSolution))
print "Length6: "+ str(len(AF.assignmentFunction([1,1,0],1).finalSolution))
print "Length7: "+ str(len(AF.assignmentFunction([0,1,1],1).finalSolution))
print "Length8: "+ str(len(AF.assignmentFunction([1,1,1],1).finalSolution))

#print AF.assignmentFunction([1,1,1],2)
#print AF.assignmentFunction([1,1,1],3)
