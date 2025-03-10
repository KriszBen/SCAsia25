import numpy as np
import networkx as nx
from threading import Thread
import math
import timeit
import pyomo.environ as pyo
import copy
from pulser import Register
from pulser import Pulse, Sequence, Register
from pulser.devices import AnalogDevice as my_device
from pulser_simulation import QutipEmulator
from pulser.waveforms import InterpolatedWaveform

def generate_first_feasible_solution(my_graph, backend = None):
    """ Give one color to each node: each color is used on only one node. Also, it gives ISs by running quantum MIS"""

    coloring_sets = {}
    #Create coloring sets with only one node
    for u in range(nx.number_of_nodes(my_graph)):
        coloring_sets[u] = [u]

    return coloring_sets
    
def solve_master_VCP(my_graph, coloring_sets, solver = 'glpk', solver_exe_path = None, solver_options = {"tmlim": 300}, tee = False, relaxation = False):
    """Reduced Master (related to the non compact formulation) for the Vertex Coloring """

    #creating the model
    my_model = pyo.ConcreteModel("master") # model
    my_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    #Creating the sets
    my_model.V = pyo.Set(initialize = list(my_graph.nodes())) #set of vertices
    my_model.S = pyo.Set(initialize = [key for key in coloring_sets.keys()] ) #set of edges

    #Creating the variables
    if relaxation == True:
        my_model.x = pyo.Var(my_model.S, domain=pyo.NonNegativeReals,bounds=(0,1), initialize = True)# holds 1 iff node u is in the cut-set; 0 otherwise
    else:
        my_model.x = pyo.Var(my_model.S, domain=pyo.Binary, initialize = True)# holds 1 iff node u is in the cut-set; 0 otherwise

    #Each vertex must be in at least one set
    def const(self,u):
        return sum(my_model.x[s] for s in my_model.S if u in coloring_sets[s]) == 1.0 #Constraint number 6 on the paper
    my_model.const = pyo.Constraint(my_model.V, rule=const)


    #Objective Funtion --> minimize the number of colors used to cover all nodes
    my_model.obj = pyo.Objective(rule=sum(my_model.x[s] for s in my_model.S),sense=pyo.minimize)
    
    #solving model
    if solver_exe_path !=None:
        results = pyo.SolverFactory(solver, executable = solver_exe_path).solve(my_model, options = solver_options, tee = tee)
    else:
        results = pyo.SolverFactory(solver).solve(my_model, options = solver_options, tee = tee)

    #geting the results
    coloring = {}
    for s in my_model.S:
        if pyo.value(my_model.x[s])  >= 10e-3:
            for n in coloring_sets[s]:
                coloring[n] = s

    #returning the solution
    if relaxation == True:
        return coloring, [my_model.dual[my_model.const[pi]] for pi in my_model.V ] #pi == w on the paper
    else:
        return coloring

   
class PulserSimLocal():
    @classmethod
    def run(cls, seq, N_samples=100):
        simul = QutipEmulator.from_sequence(seq)

        #Run the simumation
        results = simul.run()
    
        #Get the final samplings
        final = results.get_final_state()
        return results.sample_final_state(N_samples= N_samples)


def solve_quantum_MIS(my_graph, duals = None, all_ISs = False, new_Omegas = False, backend = None):
    """
    Quantum sampler
    Solve the MIS for a given graph
    """
    #If there is only one node, no need to solve the problem
    if nx.number_of_nodes(my_graph) == 1:
        if type(duals) != type(None):
            IS = list(my_graph.nodes())
            if  all_ISs == True:
                return {0 : {'IS': IS, 'reduced cost' : duals[IS[0]]}}
            else:
                return IS, duals[IS[0]]
        else:
            IS = list(my_graph.nodes())
            if  all_ISs == True:
                return {0 : {'IS': IS, 'reduced cost' : len(IS)}}
            else:
                return IS, len(IS)

    elif nx.density(my_graph) == 0:
        if type(duals) != type(None):
            IS = [node for node in range(len(duals)) if duals[node] > 0]
            if  all_ISs == True:
                return {0 : {'IS': IS, 'reduced cost' : sum([duals[dual] for dual in range(len(duals)) if duals[dual] > 0 ])}}
            else:
                return IS, sum([duals[dual] for dual in range(len(duals)) if duals[dual] > 0 ])
        else:
            IS = list(my_graph.nodes())
            if  all_ISs == True:
                return {0 : {'IS': IS, 'reduced cost' : len(IS)}}
            else:
                return IS, len(IS)

    elif nx.density(my_graph) == 1:
        if type(duals) != type(None):
            IS = [duals.index(max(duals))]
            if  all_ISs == True:
                return {0 : {'IS': IS, 'reduced cost' : duals[IS[0]]}}
            else:
                return IS, duals[IS[0]]
        else:
            IS = [list(my_graph.nodes())[0]]
            if  all_ISs == True:
                return {0 : {'IS': IS, 'reduced cost' : 1}}
            else:
                return IS, 1
    
    #get the register and pulse parameters calculated in advance
    
    layout_info  = my_graph.graph['quantum_param']['register']._layout_info
    traps = []
    keys = []
    for key in my_graph.graph['quantum_param']['register'].qubits:
        if key in my_graph.nodes():
            traps.append(layout_info.trap_ids[key])
            keys.append(key)
            # reg[key] = my_graph.graph['quantum_param']['register'].qubits[key]
    reg = my_graph.graph['quantum_param']['layout'].define_register(*traps, qubit_ids=keys)

    try:
        if new_Omegas == False:
            Omega = my_graph.graph['quantum_param']['sequence']["Omega"]
        else:
            Omega = get_new_Omega(my_graph, reg.qubits)

        delta = my_graph.graph['quantum_param']['sequence']["delta"]
        T = my_graph.graph['quantum_param']['sequence']["ev_time"]

    except:
        print("bug")
        Omega = 10
        delta = 3
        T = 6000

    #Create the pulse with the parameters
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [-delta, 0 , delta]),
        0,
    )
    #Create the sequence
    seq = Sequence(reg, my_device)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")
    seq.draw()
    seq.register.draw()
    
    count_dict_place = backend.run(seq, N_samples = 100)
    count_dict = PulserSimLocal().run(seq, N_samples = 100)


    #Calculate the cost of each sampled bistring
    for key in count_dict.keys():
        count_dict[key] = get_cost_MIS(key, my_graph, penalty=100, duals = duals)

    if all_ISs == False:
        #get the best one, i.e., the largest IS
        best_string = np.array(list(max(count_dict, key = count_dict.get)),dtype=int)
        IS = []
        if count_dict[''.join(str(i) for i in best_string)] > 0 : # this condition  is not needed 
            for qubit in range(len(best_string)):
                if best_string[qubit] == 1:
                    IS.append(reg.qubit_ids[qubit])

        return IS, count_dict[''.join(str(i) for i in best_string)]
    else:
        #get all feasible ones
        ISset = {}
        set = 0 #set index as auxiliar variable
        for key in count_dict.keys():
            if count_dict[key] > 0 : #only if the bitstring is feasible
                IS = []
                bstring = np.array(list(key),dtype=int)
                for qubit in range(len(bstring)):
                    if bstring[qubit] == 1:
                        IS.append(reg.qubit_ids[qubit])
                ISset[set] = {'IS': IS, 'reduced cost' : count_dict[key]}# reduced cost  = IS size
                set+=1 #update the set index of the last one

        return ISset

def get_new_Omega(my_graph, positions):
    """
    Calculate the new omega based on the distance of the remaining vertices
    eq19 eq20 on the paper
    """
    dc = []
    for e in my_graph.edges():
        dc.append(math.dist(positions[e[0]], positions[e[1]]))
    
    dd = []
    for e in nx.complement(my_graph).edges():
        dd.append(math.dist(positions[e[0]], positions[e[1]]))

    if len(dc)*len(dd)> 0:
        return min(15,my_device.interaction_coeff/(min(max(dc),min(dd))**6))
    elif len(dc)> 0:
        return min(15,my_device.interaction_coeff/(max(dc)**6))
    else:
        return min(15,my_device.interaction_coeff/(min(dd)**6))

def get_cost_MIS(bitstring, my_graph, penalty=10e3, duals = None):
    """Calculate the IS cost of a given bitstring"""

    # my_graph = igraph.Graph.from_networkx(my_graph)
    
    try: #if the bitstring is a string variable
        z = np.array(list(bitstring), dtype=int)
    except: #if the bistring is a 0-1 vector
        z = bitstring

    if type(duals) == type(None): #if nodes has no weights, we set 1 to all of them
        duals = list(np.ones(len(z)))

    #Getting the adjacency matrix of the graph
    # A = np.array(my_graph.get_adjacency().data)
    A = nx.to_numpy_array(my_graph)

    #Calculating the cost:
    cost =  sum(duals[i]*z[i] for i in range(len(z))) -  penalty*(z.T @ np.triu(A) @ z)

    return max(0,cost)
 
def solve_quantum_CG(my_graph, solver_exe_path = None, allISs = True, new_Omega = True, backend = None):
    """ 
    Flowchart Fig3 on the paper
    Hybrid classical-quantum Column Generation: classical reduced master and quantum pricing"""

    #Get the first feasible solution with some IS sets"""
    coloring_sets = generate_first_feasible_solution(my_graph, backend = backend)

    #auxiliar variales
    test = False
    sol_ev = [] # will hold all the number of colors used on each of itiration indexes
    shots = []
    rcs = [] #will hold the reduced costs from each iteration
    last_sol = 2*nx.number_of_nodes(my_graph) #upper bound
    counter = 0  #used to in the stop criterium
    ctr = 0
    my_graph.graph["dual solution quantum CG"] = [] #will store the values of the dual varaibles after solving each instance of the master problem

    start = timeit.default_timer()
    while test == False:#while there's still room for improvement
        shots.append(len(shots)+1)
        
        #solving master classicaly
        sol_exp, pi = solve_master_VCP(my_graph, coloring_sets, solver_exe_path = solver_exe_path, relaxation=True)
        #saving the current solution's value
        my_graph.graph["dual solution quantum CG"].append(pi) #not neeeded
        sol_ev.append(len(np.unique([sol_exp[key] for key in sol_exp.keys()])))

        #Create a auxiliar graph with only nodes that have a positive weight
        H = copy.deepcopy(my_graph)
        H.remove_nodes_from([node for node in my_graph.nodes() if pi[node] < 10e-3])
        node_weights = [pi[node] for node in my_graph.nodes() if pi[node] > 10e-3 ]

        new_sets = solve_quantum_MIS(H, duals = node_weights, all_ISs = allISs, new_Omegas = new_Omega, backend = backend)

        #if treating a possible different output from quantum MIS function
        if allISs == False:
            new_sets = {0: {'IS': new_sets[0], 'reduced cost': new_sets[1]}}

        #If at least one set was generated
        if len(new_sets.keys()) > 0 and counter < 2:
            redCost = []
            #For each weighted IS found
            for key in new_sets.keys():
                if new_sets[key]['reduced cost'] > 1.0: #only with sets that can deacread the value of the cost function : verification of condition eq18 on the paper
                    #updating set of colors
                    redCost.append(new_sets[key]['reduced cost'])
                    coloring_sets[len(coloring_sets.keys())] = new_sets[key]['IS']

            if len(redCost) > 0 and max(redCost) > 1.0:#if at least one set the can decrease the number of colors is found
                rcs.append(max(redCost))
                
                #If the coloring was improved
                if last_sol > sol_ev[-1]:
                    last_sol = sol_ev[-1]
                    counter = 0

                #if no improvement is made (last_sol  == sol_ev[-1]), then update the counter and try again (max tries = 5)
                else:
                    counter+=1 #if there's no change in the cost function (meaning that we creat a symmetric solution )
            else:
                #no interesting set was found
                test = True 

        #elif counter > 3: 
           # test = True
        else:#no set was found
            test = True
    #solving the final ILP, setting the varibales to Binary values only
    sol_exp = solve_master_VCP(my_graph, coloring_sets, solver = 'glpk', solver_exe_path = solver_exe_path, solver_options = {"tmlim": 300}, tee = False, relaxation=False)
    end = timeit.default_timer()

    my_graph.graph['quantum pricing iteration'] =  shots
    my_graph.graph['quantum CG coloring'] = sol_ev
    my_graph.graph['quantum reduced costs'] = rcs
    my_graph.graph['quantum CG runtime'] = end-start
    my_graph.graph['quantum sets'] = coloring_sets
    my_graph.graph['quantum CG binary solution'] = sol_exp
    