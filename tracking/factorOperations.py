# factorOperations.py
# -------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()



def joinFactors(factors: List[Factor]):
    

    factors_list = list(factors)

    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors_list]
    if len(factors_list) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factors_list[-1]) # Or handle factor identification differently
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors_list)))


    if not factors_list:
        
        return None 

    all_variables = set()
    all_unconditioned = set()
    all_conditioned = set()
    variable_domains = factors_list[0].variableDomainsDict() 

    for factor in factors_list:
        all_variables.update(factor.variablesSet())
        all_unconditioned.update(factor.unconditionedVariables())
        all_conditioned.update(factor.conditionedVariables())

   
    new_conditioned_variables = all_conditioned - all_unconditioned
    new_unconditioned_variables = all_variables - new_conditioned_variables

    new_factor = Factor(new_unconditioned_variables, new_conditioned_variables, variable_domains)

    for assignment_dict in new_factor.getAllPossibleAssignmentDicts():
        probability = 1.0
        for factor in factors_list:
            probability *= factor.getProbability(assignment_dict)
        
        new_factor.setProbability(assignment_dict, probability)

    return new_factor


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
       
        
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

       
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        
        new_unconditioned_variables = factor.unconditionedVariables() - {eliminationVariable}
        new_conditioned_variables = factor.conditionedVariables()
        variable_domains = factor.variableDomainsDict()

        new_factor = Factor(new_unconditioned_variables, new_conditioned_variables, variable_domains)

        for assignment_dict in new_factor.getAllPossibleAssignmentDicts():
            summed_probability = 0.0
            for elimination_value in variable_domains[eliminationVariable]:
                full_assignment = assignment_dict.copy()
                full_assignment[eliminationVariable] = elimination_value
                summed_probability += factor.getProbability(full_assignment)
            
            new_factor.setProbability(assignment_dict, summed_probability)
            
        return new_factor
        

    return eliminate

eliminate = eliminateWithCallTracking()

