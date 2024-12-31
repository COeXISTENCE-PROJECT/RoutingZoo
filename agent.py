import numpy as np
import random

from abc import ABC, abstractmethod


"""
IMPROVEMENTS:

1. have a given number number of paths
2. each agent has different choice set (of a fixed size)
3. each agent has own parameters
4. add information to choice (previous days)
"""

class Agent(ABC):

    """
    This is an abstract class for agents, to be inherited by specific type of agent classes
    It is not to be instantiated, but to provide a blueprint for all types of agents
    """
    
    def __init__(self, id, start_time, origin, destination):
        self.id = id
        self.start_time = start_time
        self.origin = origin
        self.destination = destination

    @abstractmethod
    def act(self, state):  
        # Pick action according to your knowledge, or randomly
        pass

    @abstractmethod
    def learn(self, action, reward, observation):
        # Pass the applied action and reward once the episode ends, and it will remember the consequences
        pass




class HumanAgent(Agent):

    def __init__(self, id, start_time, origin, destination, params, initial_knowledge, action_space_size, **kwargs,):
        super().__init__(id, start_time, origin, destination)

        #simulation-params

        self.kind = 'Human'
        self.action_space = action_space_size
        self.stored_utilities = list(np.zeros(self.action_space,dtype=int))
        self.last_reward = None


        #Cost-params
        self.remember = int(kwargs['remember'])
        self.gamma_c = kwargs['gamma_c']
        self.old_c_hat = [0,0]
        self.cost = np.array(initial_knowledge, dtype=float)
        self.days = [[] for _ in range(action_space_size)]

        #Utility-params
        self.type = kwargs['learning_type']
        self.beta = params['beta']
        self.gamma_u = kwargs['gamma_u']
        self.greedy = kwargs['greedy']

        #Noise-params
        self.noise_alpha = kwargs['noise_alpha']
        self.stored_noises = None
        self.human_noise = np.random.gumbel() #RK: This shall be generalized towards 'k'-paths noise at init.
        self.noise_taste = kwargs['noise_taste']
        self.noise_random = kwargs['noise_random']

        #Alpha values base on learning type

        if self.type == 'culo': #this is redundant, this can be hanlded directly at the initialization via kc.PARAMS, no need to do it here
            self.alpha_zero = 1
            self.alpha_j = params['alpha_j']
        elif self.type == 'markow':
            self.alpha_zero = params['alpha_zero']
            self.alpha_j = 1 - self.alpha_zero
        elif self.type == 'weight':
            self.alpha_zero = 0
            self.alpha_j = params['alpha_j']
        else:
            print('Correct model definition missing - default markow')
            self.alpha_zero = params['alpha_zero']
            self.alpha_j = 1 - self.alpha_zero


        #Creating memory for Weigted model
        self.create_memory()

        #Create noise for routes
        self.route_taste = self.taste_noise()
    
    def act(self, state):  
        """ 
        the implemented dummy logit model for route choice, make it more generate, calculate in graph levelbookd
        """

        self.make_noise()

        utilities = list(map(lambda x,y: ((self.beta * x) + y), self.cost,self.stored_noises)) #RK: here to do: self.noise[k]

        if abs(self.old_c_hat[-2] - self.old_c_hat[-1]) > self.gamma_u:

            p = np.random.random()

            if p < self.greedy:

                action = np.random.randint(low=0,high=self.action_space) #RK: First bounded rationality, then greedy.

            else:

                action = np.argmin(utilities)
                self.stored_utilities = utilities
        
        else:

            action = np.argmin(self.stored_utilities)


        return action       


    def learn(self, action, observation):

        reward = self.get_reward(observation)
        self.last_reward = reward

        if abs(self.old_c_hat[-1]-reward) > self.gamma_c:

            if self.type == 'weight':
                c_hat = 0
                if self.cost[action] != reward: #RK: This is quite dangerous assumption. you beeter check this condition somehow else (what if the costs are the same on two paths?)
                    del(self.days[action][len(self.days[action])-1]) #RK: I would try to data structure called queue here https://docs.python.org/3/library/queue.html
                    self.days[action].insert(0,self.cost[action])
                for memory in range(self.remember): #memory

                    c_hat += self.alpha_j[memory] * self.days[action][memory]

            else:

                    c_hat = self.alpha_zero * self.cost[action] + self.alpha_j * reward

            self.cost[action] = c_hat
            self.old_c_hat.append(c_hat)


    def get_reward(self, observation: list[dict]) -> float:
        """ This function calculated the reward of each individual agent. """
        own_tt = -1 * next(obs['travel_time'] for obs in observation if obs['id'] == self.id) ## Anastasia added the -1
        return own_tt
    
    #Create noises
    def taste_noise(self):

        taste_noise = np.random.gumbel(size=self.action_space)
        
        return taste_noise

    def make_noise(self):

        noises = self.noise_alpha * self.human_noise + (self.noise_taste) * self.route_taste + self.noise_random * np.random.gumbel(size=self.action_space) #RK: This shall be generalized to noise for each actin for each day
        
        self.stored_noises = noises

    #Create memory for weighted model
    
    def create_memory(self): #RK: are you sure this shall be in init?

        for i in range(self.action_space):
            for r in range(self.remember):
                self.days[i].append(self.cost[i])
