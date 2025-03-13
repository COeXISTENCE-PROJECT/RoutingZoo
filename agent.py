import numpy as np
import random
from collections import deque


from abc import ABC, abstractmethod
from learning_model import BaseLearningModel



class GeneralHuman(BaseLearningModel):
    """
    This is the general model of human route-choice behaviour which can accomodate several classic methods.
    It follows the two-step structure of `learn` and `act`.
    The new experience after travelling is used to update expected costs in the `learn`.
    The learned experiences is used to make routing decision (action) in `act`.

    * The variability (non-determinism) can be controlled at several levels:
    - epsilon_i_variability, epsilon_k_i_variability, epsilon_k_i_t_variability - variance of normal distriburion from which error terms are drawn
    - beta_k_i_variability - variance of normal distribution for which \beta_k_i is drawn (computed in utility)
    - noise_weight_agent, noise_weight_path, noise_weight_day - relative weights for the error term composition

    * The parameters of the model are:
    - beta              -   negative value!!! - multiplier of reward (travel time) used in utility, determines sensitivity
    - greedy            -   1 - exploration_rate , probability with which the choice are rational (argmax(U)) and not probabilistic (random)
    - gamma_u, gamma_c  -   bounded rationality components. Expressed as relative increase in costs/utilities below which user do not change behaviour (do not notice it)
    - alpha_zero        -   weight witch which the recent experience is carried forward in learning
    - remember          -   number of days remembered to learn
    - alphas            -   vector of weights for historically recorded reward in weighted average !!needs to be size of 'remember'!!
    
    """

    def __init__(self, params, initial_knowledge):
        """
        params is the dictionary, exaclty like the one used in RouteRL
        
        initial knowledge shall come from SUMO - free flow travel time (or utilities)
        """
        super().__init__()
        self.costs = np.array(initial_knowledge, dtype=float) # cost matrix
        self.action_space = len(self.costs) # number of paths
        self.mean_time = np.mean(self.costs) # mean free flow travel time over paths - possibly useful to normalize errors
        
        #weights of respective componenets of error term (should sum to one)
        self.noise_weight_agent = params.get("noise_weight_agent",0.2) #drawn once per agent
        self.noise_weight_path = params.get("noise_weight_path",0.6) # drawn once per agent per path
        self.noise_weight_day = params.get("noise_weight_path",0.2) # drawn daily per agent per path
        assert self.noise_weight_agent + self.noise_weight_agent + self.noise_weight_day  == 1 , "relative weights in error terms do not sum up to 1"

        # \beta_{t,k,i} agent-specific travel time multiplier per path
        self.beta_k_i = params.get("beta",-1)*np.random.normal(1,params.get("beta_k_i_variability",0), size = self.action_space) # used to compute Utility in `act`
        self.random_term_agent = np.random.normal(0, params.get("epsilon_i_variability",0)) # \vareps_{i}
        self.random_term_path = np.random.normal(0,params.get("epsilon_k_i_variability",0), size = self.action_space) # \vareps_{k,i}
        self.random_term_day = params.get("epsilon_k_i_t_variability",0) # \vareps_{k,i,t}

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
        """
        update 'costs' of 'action' after receiving a 'reward'
        """
        self.memory[action].append(reward) #add recent reward to memory (of rewards)
      
        if abs(self.costs[action]-reward)/self.costs[action]>=self.gamma_c: #learn only if relative difference in rewards is above gamma_c
            weight_normalization_factor = 1/(self.alpha_zero+ sum([self.alphas[i] for i,j in enumerate(self.memory[action])])) # needed to make sure weights are alsways summed to 1
            self.costs[action] = weight_normalization_factor * self.alpha_zero* self.costs[action] #experience weights
            self.costs[action] += sum([weight_normalization_factor * self.alphas[i]*self.memory[action][i] for i,j in enumerate(self.memory[action])]) # weighted average of historical rewards
   

    def act(self, state):  
        """
        select path from action space based on learned expected costs"""
        # for each path you multiply the expected costs with path-specific beta (drawn at init) and add the noise (computed from 3 components in `get_noises`)
        utilities = [self.beta_k_i[i]* self.costs[i] + noise for i,noise in enumerate(self.get_noises())] 
        if self.first_day or abs(self.last_action["utility"] - utilities[self.last_action['action']])/self.last_action["utility"] >= self.gamma_u: #bounded rationality
            if np.random.random() < self.greedy:
                action = int(np.argmax(utilities)) # greedy choice 
            else:
                 action = np.random.choice(self.action_space)  # random choice
        else:
            action = self.last_action['action']    
        self.first_day = False
        self.last_action = {"action":action, "utility":utilities[action]}
        return action       
    
    def get_noises(self):
        """"
        compute random term for the utility, composed of 3 parts - two drawn at init and one here, inside

        returns vector of errors
        """
        daily_noise = np.random.normal(0,self.random_term_day, size= self.action_space)
        return [self.noise_weight_agent * self.random_term_agent + 
                self.noise_weight_path * self.random_term_path[k] + 
                self.noise_weight_day * daily_noise[k]
                    for k,_ in enumerate(self.costs)]
    

"""
Library of specific models implementations
"""
class ProbabilisticModel(GeneralHuman):
     # random choice
     def __init__(self, params, initial_knowledge):
        params['greedy']= 0 
        super().__init__(params, initial_knowledge)

class GawronModel(GeneralHuman):
    # classic 0.8 - 0.2 exponential smoothing/markov/gawron
    def __init__(self, params, initial_knowledge):
        params['remember']= 1 
        params['alpha_zero'] = 0.2
        params['alphas'] = [0.8]
        params['greedy']= 1
        super().__init__(params, initial_knowledge)

class WeightedModel(GeneralHuman):
    #5-day weighted average with decreasing weights
    def __init__(self, params, initial_knowledge):
        params['remember']= 5 
        params['alpha_zero'] = 0
        params['alphas'] = [1/(n+2)/1.45 for n in range(5)]
        params['greedy']= 1
        super().__init__(params, initial_knowledge)

