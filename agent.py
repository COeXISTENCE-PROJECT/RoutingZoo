import numpy as np
import random

from abc import ABC, abstractmethod

from keychain import Keychain as kc


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

    def __init__(self, id, start_time, origin, destination, params, initial_knowledge, action_space_size, human_noise, mutate_to=None):
        super().__init__(id, start_time, origin, destination)

        self.kind = kc.TYPE_HUMAN
        self.mutate_to = mutate_to
        self.action_space = action_space_size
        self.stored_utilities = list(np.zeros(self.action_space,dtype=int)) #RK: This shall be generic, not 3 elements (np.zeros(kc.ACTION_SPACE) - or sth like this)
        self.act_type = params[kc.ACT_TYPE]
        self.type = params[kc.LEARNING_TYPE]
        #self.alpha_nulla = params[kc.ALPHA_NULLA]
        #self.alpha_j = params[kc.ALPHA_J] #RK: How about varying weights for weighted average?
        self.remember = params[kc.REMEMBER]
        self.delta = params[kc.DELTA]
        self.greedy = params[kc.GREEDY]
        self.test = id
        self.old_c_hat = [0]

        if self.type == 'culo': #this is redundant, this can be hanlded directly at the initialization via kc.PARAMS, no need to do it here
            self.alpha_nulla = 1
            self.alpha_j = params[kc.ALPHA_J]
        elif self.type == 'markow':
            self.alpha_nulla = params[kc.ALPHA_NULLA]
            self.alpha_j = 1 - self.alpha_nulla
        elif self.type == 'weight':
            self.alpha_nulla = 0
            self.alpha_j = params[kc.ALPHA_J]
        else:
            print('Correct model definition missing - default markow')
            self.alpha_nulla = params[kc.ALPHA_NULLA]
            self.alpha_j = 1 - self.alpha_nulla

        beta_randomness = params[kc.BETA_RANDOMNESS]
        self.beta = params[kc.BETA]
        self.alpha = params[kc.ALPHA]
        self.mu = params[kc.MU]
        self.noise_alpha = params[kc.NOISE_ALPHA]
        self.stored_noise = None
        self.randomness = params[kc.RANDOMNES]

        self.human_noise = human_noise #RK: This shall be generalized towards 'k'-paths noise at init.

        self.cost = np.array(initial_knowledge, dtype=float)
        self.days = list(np.zeros(action_space_size))
        #for i in range(self.action_space): #RK: this shall be a param - action_space, not hard_coded to "3"
            #self.days.append([])

        self.create_memory()
        self.route_taste = self.taste_noise()

    
    def make_noise(self):

        noises = self.noise_alpha * self.human_noise + (self.noise_taste) * self.route_taste + self.noise_random + np.random.gumbel(size=self.action_space) #RK: This shall be generalized to noise for each actin for each day
        
        self.stored_noises = noises
    
    def act(self, state):  
        """ 
        the implemented dummy logit model for route choice, make it more generate, calculate in graph levelbookd
        """
        #if self.act_type == 'test':
        #    utilities = list(map(lambda x: np.exp(x * self.beta), self.cost))
        #    prob_dist = [self.calculate_prob(utilities, j) for j in range(len(self.cost))]
        #elif self.act_type == 'no_test':
        #print('before:',self.test,self.utility)
        self.make_noise()
        #noises = self.make_noise() #RK: Make noise for each path
        utilities = list(map(lambda x,y: ((self.beta * x) + y), self.cost,self.stored_noises)) #RK: here to do: self.noise[k]
        #else:
        #    raise ValueError("Define Act type")

        #self.utility = utilities
        
        #prob_dist = [self.calculate_prob(utilities, j) for j in range(len(self.cost))]
        #action = np.random.choice(list(range(len(self.cost))), p=prob_dist)
        #if self.act_type == 'test':
        #    action = np.random.choice(list(range(len(self.cost))), p=prob_dist) 
        #else:

        if abs(self.old_c_hat[-2] - self.old_c_hat[-1]) >= self.gamma_u:

            p = np.random.random()

            if p < self.greedy:

                action = np.random.randint(low=0,high=len(self.stored_utilities)) #RK: First bounded rationality, then greedy.

            else:

                action =  np.argmin(utilities)
                self.stored_utilities = utilities

            #if (self.utility.index(min(self.utility))-utilities.index(min(utilities)))**2 < self.delta:  #RK: Normalize utilities (a-b/a) and get rid of squer (unless found reference that this is advocated)

            #    action = self.utility.index(min(self.utility))
            #    self.utility = self.utility # redundant

            #else:

            #    action =  utilities.index(min(utilities))
            #    self.utility = utilities  # RK: confusing, why you mix singular (utility) with plural (utilities) - change names.
                #why here you save the latest utilites with noise and not in the case when you did not updated? Please discuss and check.

        
        #print('after:',self.test,self.utility)

        return action       


    def learn(self, action, reward, observation):
        #if self.type == 'culo': #this is redundant, this can be hanlded directly at the initialization via kc.PARAMS, no need to do it here
        #    alpha_nulla = 1
        #    alpha_j = self.alpha_j
        #elif self.type == 'markow':
        #    alpha_nulla = self.alpha_nulla #RK: Remember you call this function for every agent for every day - why do you set it here? this shall be part of init.
        #    alpha_j = 1 - alpha_nulla
        #elif self.type == 'weight':
        #    alpha_nulla = 0
        #    alpha_j = self.alpha_j
        #if self.type == 'weight':
            #if self.cost[action] != reward: #RK: This is quite dangerous assumption. you beeter check this condition somehow else (what if the costs are the same on two paths?)
            #    del(self.days[action][len(self.days[action])-1]) #RK: I would try to data structure called queue here https://docs.python.org/3/library/queue.html
            #    self.days[action].insert(0,self.cost[action])
            #for i in range(self.remember):
                    #a = (i+1)**(-1) #RK: This shall be handled via params, see at formulas in Tristan submission and implement accordingly
            #        c_hat = 0
            #        c_hat += self.alpha_j[i] * self.days[action][i] #RK: Where here is the loop over paths? you update only a single path ([action])? If so, why? let's discuss.

        #else:
            #print('Correct model definition missing - default markow')
            #alpha_nulla = self.alpha_nulla
            #alpha_j = 1 - alpha_nulla

            #c_hat = self.alpha_nulla * self.cost[action] + self.alpha_j * reward

        #if (c_hat-reward) <= self.lower or (c_hat-reward)>= self.upper:  #RK: normalize and avoid square.
        if abs(self.old_c_hat[-1]-reward) >= self.gamma_c:

            if self.type == 'weight':
                if self.cost[action] != reward: #RK: This is quite dangerous assumption. you beeter check this condition somehow else (what if the costs are the same on two paths?)
                    del(self.days[action][len(self.days[action])-1]) #RK: I would try to data structure called queue here https://docs.python.org/3/library/queue.html
                    self.days[action].insert(0,self.cost[action])
                for i in range(self.remember):
                    #a = (i+1)**(-1) #RK: This shall be handled via params, see at formulas in Tristan submission and implement accordingly
                    c_hat = 0
                    c_hat += self.alpha_j[i] * self.days[action][i]

                else:
            #print('Correct model definition missing - default markow')
            #alpha_nulla = self.alpha_nulla
            #alpha_j = 1 - alpha_nulla

                    c_hat = self.alpha_nulla * self.cost[action] + self.alpha_j * reward

            #self.cost[action] = self.cost[action] #RK: this is redundant
            self.cost[action] = c_hat
            self.old_c_hat.append(c_hat)

            #self.cost[action] = self.alpha_nulla * self.cost[action] + self.alpha_j * reward #RK: are you sure you update costs only if the path was chosen?
        #self.cost[action]=(1-self.alpha) * self.cost[action] + self.alpha * reward


    #def calculate_prob(self, utilities, n):
        #prob = utilities[n] / sum(utilities)
        #return prob
    

    def mutate(self):
        self.mutate_to.q_table = self.cost
        return self.mutate_to
    
    def create_memory(self): #RK: are you sure this shall be in init?

        for i in range(len(self.cost)):
            for r in range(self.remember):
                self.days[i].append(self.cost[i])

    def taste_noise(self):

        taste_noise = np.random.gumbel(size=self.action_space)
        
        return taste_noise





#class MachineAgent(Agent):

#    def __init__(self, id, start_time, origin, destination, params, action_space_size):
        super().__init__(id, start_time, origin, destination)

        self.kind = kc.TYPE_MACHINE

        min_alpha, max_alpha = params[kc.MIN_ALPHA], params[kc.MAX_ALPHA]
        min_epsilon, max_epsilon = params[kc.MIN_EPSILON], params[kc.MAX_EPSILON]
        min_eps_decay, max_eps_decay = params[kc.MIN_EPS_DECAY], params[kc.MAX_EPS_DECAY]

        self.epsilon = random.uniform(min_epsilon, max_epsilon)
        self.epsilon_decay_rate = random.uniform(min_eps_decay, max_eps_decay)
        self.alpha = random.uniform(min_alpha, max_alpha)
        self.gamma = params[kc.GAMMA]

        self.action_space_size = action_space_size
        # Q-table assumes only one state, otherwise should be np.zeros((num_states, action_space_size))
        self.q_table = np.zeros((action_space_size))


#    def act(self, state):
        if np.random.rand() < self.epsilon:    # Explore
            return np.random.choice(self.action_space_size)
        else:    # Exploit
            return np.argmin(self.q_table)
                

#    def learn(self, action, reward, observation):
        prev_knowledge = self.q_table[action]
        self.q_table[action] = prev_knowledge + (self.alpha * (reward - prev_knowledge))
        self.decay_epsilon()


#    def decay_epsilon(self):    # Slowly become deterministic
        self.epsilon *= self.epsilon_decay_rate