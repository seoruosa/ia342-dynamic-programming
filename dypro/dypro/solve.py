import numpy as np
import logging
import sys, time

logging.basicConfig(level=logging.ERROR)

def solve(state, numberOfStages:int, finalStateCost, decision, solveInfeasibility, 
			transitionFunction, elementaryCost, initialFMap=dict(), initialPolicy=dict(), inf=np.inf):
	""" Solves the optimality recursive equation using a backward strategy

	Parameters

	state : function 
			Returns all possible states for each stage
	
	numberOfStages : int
			Number of stages
	
	finalStateCost : function 
			Returns the cost of all final states
	
	decision : function
			Returns all possible decisions for each stage
	
	solveInfeasibility : function of (k, xk_next, FMap)
			Returns (feasible xk_next, _F)
	
	transitionFunction : function of (k, xk, uk)
			Returns the next state
	
	elementaryCost : function of (k, xk, uk)
			Returns the cost of decision on state
	
	initialMap : dict

	initialPolicy : dict

	Returns:
		[(Map of costs : dict, policy : dict))]: [description]
	"""    

	INFEASIBLE_MAP = dict()

	FMap = dict.copy(initialFMap)
	policy = dict.copy(initialPolicy)

	for xk in state(numberOfStages):
		FMap[numberOfStages, xk] = finalStateCost(xk)
		logging.debug(f"{FMap}")
	
	for k in range(numberOfStages-1, -1, -1): #n-1 to 0
		start_time = time.time()
		for xk in state(k):
			F_aux = inf
			u_aux = None
			
			# Calculates the best decision on stage k and state uk
			for uk in decision(k):
				xk_next_maybe_infeasible = transitionFunction(k, xk, uk)
				

				if (k+1, xk_next_maybe_infeasible) not in FMap and (k+1, xk_next_maybe_infeasible) not in INFEASIBLE_MAP:
					xk_next, _F = solveInfeasibility(k+1, xk_next_maybe_infeasible, FMap)
					INFEASIBLE_MAP[k+1, xk_next_maybe_infeasible] = (xk_next, _F)

				elif (k+1, xk_next_maybe_infeasible) in INFEASIBLE_MAP:
					xk_next, _F = INFEASIBLE_MAP[k+1, xk_next_maybe_infeasible]

				else:
					xk_next, _F = (xk_next_maybe_infeasible, FMap[k+1, xk_next_maybe_infeasible])

				# F_aux_uk = elementaryCost(k, xk, uk) + FMap[k+1, xk_next]
				F_aux_uk = elementaryCost(k, xk, uk) + _F

				logging.debug(f"{k} {xk} {uk} {xk_next_maybe_infeasible} next: {xk_next} {F_aux_uk} {elementaryCost(k, xk, uk)}")


				if F_aux > F_aux_uk:
					F_aux = F_aux_uk
					u_aux = uk
			
			FMap[k, xk] = F_aux
			policy[k, xk] = u_aux
		print(f"Stage {k} elapsed {time.time() - start_time} sec")
	
	return (FMap, policy)



	"""
	proximo passo infactivel
	
	em k tenho demanda=10, estoque=5, produzi=4 
	=> 
	em k + 1  -> terei no estoque -1 item
	<=>
	eu vou ter que comprar 1 item -----> custoEstoque(5) + custoProducao(4) + custoInfactibilidade(xk_next=-1)
	=> 
	em k + 1 terei demanda=..., estoque=0, produzo=...
	"""


def solveStochastic(state, numberOfStages:int, finalStateCost, decision, realizableRandomValues,  solveInfeasibility, 
			transitionFunction, elementaryCost, initialFMap=dict(), initialPolicy=dict(), inf=np.inf):
	""" Solves the optimality recursive equation using a backward strategy

	Parameters

	state : function 
			Returns all possible states for each stage
	
	numberOfStages : int
			Number of stages
	
	finalStateCost : function 
			Returns the cost of all final states
	
	decision : function
			Returns all possible decisions for each stage
	
	solveInfeasibility : function of (k, xk_next, FMap)
			Returns (feasible xk_next, _F)
	
	transitionFunction : function of (k, xk, uk)
			Returns the next state
	
	elementaryCost : function of (k, xk, uk)
			Returns the cost of decision on state
	
	initialMap : dict

	initialPolicy : dict

	Returns:
		[(Map of costs : dict, policy : dict))]: [description]
	"""    

	FMap = dict.copy(initialFMap)
	policy = dict.copy(initialPolicy)

	for xk in state(numberOfStages):
		FMap[numberOfStages, xk] = finalStateCost(xk)
		logging.debug(f"{FMap}")
	
	for k in range(numberOfStages-1, -1, -1): #n-1 to 0
		start_time = time.time()

		for xk in state(k):
			F_aux = inf
			u_aux = None

			# Calculates the best decision on stage k and state uk
			for uk in decision(k):
				
				# contains the expectation of the cost
				E_F_aux = 0.0 

				for wk in realizableRandomValues(k).randomValueIterator():
					xk_next_maybe_infeasible = transitionFunction(k, xk, uk, wk.getValue())
					

					if (k+1, xk_next_maybe_infeasible) not in FMap:
						xk_next, _F = solveInfeasibility(k+1, xk_next_maybe_infeasible, FMap)
					else:
						xk_next, _F = (xk_next_maybe_infeasible, FMap[k+1, xk_next_maybe_infeasible])

					E_F_aux += (elementaryCost(k, xk, uk) + _F)*wk.getProbability()

					# logging.debug(f"{k} {xk} {uk} {xk_next_maybe_infeasible} next: {xk_next} {F_aux_uk} {elementaryCost(k, xk, uk)}")


				if F_aux > E_F_aux:
					F_aux = E_F_aux
					u_aux = uk

			FMap[k, xk] = F_aux
			policy[k, xk] = u_aux
		print(f"Stage {k} elapsed {time.time() - start_time} sec")
	
	return (FMap, policy)

def optimalTrajectory(initialState, numberOfStages, policy, transitionFunction):
		"""
		Calculates the optimal trajectory given a initial state

		Args:
			initialState (np.array): initial state to calculate the optimal trajectory

		Returns:
			u_optimal: states of the optimal trajectory
			policy_optimal: decisions for the optimal trajectory
		"""
		u_optimal = [initialState]
		policy_optimal = list()
		for k in range(numberOfStages):
			policy_optimal.append(policy[k, u_optimal[-1]])
			u_optimal.append(transitionFunction(k, u_optimal[-1], policy_optimal[-1]))
		
		return (u_optimal, policy_optimal)