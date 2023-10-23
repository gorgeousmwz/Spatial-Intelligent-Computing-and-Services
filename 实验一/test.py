from code import fitness_function

def coding(individual):
    t=[]
    for i in individual:
        t.append(i[0]*70+i[1])
    return t

coder=coding([[49,40],[50,20],[49,47],[18,51],[33,68]])
print(coder)
print(fitness_function(coder))