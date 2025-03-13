import numpy as np
import random
from collections import deque


from abc import ABC, abstractmethod
from learning_model import BaseLearningModel


"""
IMPROVEMENTS:

1. have a given number number of paths
2. each agent has different choice set (of a fixed size)
3. each agent has own parameters
4. add information to choice (previous days)
"""



class GeneralHuman(BaseLearningModel):

    def __init__(self, params, initial_knowledge):
        super().__init__()

        self.costs = np.array(initial_knowledge, dtype=float) # cost matrix
        self.action_space = len(self.costs) # number of paths
        
        #weights of respective componenets of error term (should sum to one)
        self.noise_weight_agent = params.get("noise_weight_agent",0.2) #drawn once paer agent
        self.noise_weight_path = params.get("noise_weight_path",0.6) # drawn once per agent per path
        self.noise_weight_day = params.get("noise_weight_path",0.2) # drawn daily per agent per path
        assert self.noise_weight_agent + self.noise_weight_agent + self.noise_weight_day  , "relative weights in error terms do not sum up to 1"

        self.beta_k_i = params.get("beta",-1)*np.random.normal(1,params.get("beta_k_i_variability",0), size = self.action_space) # \beta_{t,k,i} agent-specific travel time multiplier per paht

        self.random_term_agent = np.random.normal(0, params.get("beta_i_variability",0)) # \vareps_{i}
        self.random_term_path = np.random.normal(0,params.get("beta_k_i_variability",0), size = self.action_space) # \vareps_{k,i}
        self.random_term_day = params.get("beta_k_i_t_variability",0) # \vareps_{k,i,t}

        self.greedy = params.get("greedy",1) # probability that agent will make a greedy choice (and not random exploration)

        self.gamma_c = params.get('gamma_c',0) #bounded rationality on costs (relative)
        self.gamma_u = params.get('gamma_u',0) #bounded rationality on utilities (relative)
        self.remember = int(params.get('remember',1)) #window for averaging out

        self.alpha_zero = params.get('alpha_zero',0.2) #weight of recent experience in learning

        self.alphas = params.get('alphas',[0.8]) #weights for the memory in respective days
        assert len(self.alphas)==self.remember , "weights of history $\alpha_i in 'alphas' and 'remember do not match"
        assert abs(self.alpha_zero + sum(self.alphas) - 1)<0.01 , "weights for weighted average do not sum up to 1"
        
        self.memory = [deque([cost],maxlen=self.remember) for cost in self.costs] # memories of experiences per path
        
        self.first_day = True

    def learn(self, state, action, reward):
      
        if abs(self.costs[action]-reward)/self.costs[action]>=self.gamma_c: #learn only if relative difference is above gamma_c
            weight_normalization_factor = 1/(self.alpha_zero+ sum([self.alphas[i] for i,j in enumerate(self.memory[action])]))
            self.costs[action] = weight_normalization_factor * self.alpha_zero* reward #experience
            self.costs[action] += sum([weight_normalization_factor * self.alphas[i]*self.memory[action][i] for i,j in enumerate(self.memory[action])]) # weighted average of history
        self.memory[action].append(reward) #add recent reward to memory


        
    
    def act(self, state):  
        utilities = [self.beta_k_i[i]* self.costs[i] + noise for i,noise in enumerate(self.get_noises())]

        if self.first_day or abs(self.last_action["utility"] - utilities[self.last_action['action']])/self.last_action["utility"] >= self.gamma_u: #bounded rationality

            if np.random.random() < self.greedy:
                action = int(np.argmax(utilities)) # greedy choice 
            else:
                 action = int(np.random.choice(self.action_space))  # random choice
        else:
            action = self.last_action['action']    
        self.first_day = False
        self.last_action = {"action":action, "utility":utilities[action]}
        return action       
    
    def get_noises(self):
            daily_noise = np.random.normal(0,self.random_term_day, size= self.action_space)
            return [self.noise_weight_agent * self.random_term_agent + 
                    self.noise_weight_path * self.random_term_path[k] + 
                    self.noise_weight_day * daily_noise[k]
                     for k,_ in enumerate(self.costs)]
    


class ProbabilisticModel(GeneralHuman):
     def __init__(self, params, initial_knowledge):
        params['greedy']= 0 
        super().__init__(params, initial_knowledge)

class GawronModel(GeneralHuman):
    def __init__(self, params, initial_knowledge):
        params['remember']= 1 
        params['alpha_zero'] = 0.2
        params['alphas'] = [0.8]
        params['greedy']= 1
        super().__init__(params, initial_knowledge)

class WeightedModel(GeneralHuman):
    def __init__(self, params, initial_knowledge):
        params['remember']= 5 
        params['alpha_zero'] = 0.5
        params['alphas'] = [1/(n+3) for n in range(params['remember'])]
        weight_normalization_factor = 1/(0.5+ sum([params['alphas'][i] for i in range(params['remember'])]))
        params['alpha_zero'] = weight_normalization_factor * 0.5
        params['alphas']=[weight_normalization_factor*_ for _ in params['alphas']]
        params['greedy']= 1
        super().__init__(params, initial_knowledge)

