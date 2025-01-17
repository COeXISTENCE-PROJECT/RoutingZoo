import numpy as np
import pandas as pd
import random

from prettytable import PrettyTable

from agent import MachineAgent, HumanAgent
from keychain import Keychain as kc
from utilities import make_dir


def create_agent_objects(params, free_flow_times):

    """
    Generates agent objects
    """

    # Getting parameters
    num_agents = params[kc.NUM_AGENTS]
    ratio_mutating = params[kc.RATIO_MUTATING]
    simulation_timesteps = params[kc.SIMULATION_TIMESTEPS]
    num_origins = len(params[kc.ORIGINS])
    num_destinations = len(params[kc.DESTINATIONS])

    agent_attributes = params[kc.AGENT_ATTRIBUTES]
    action_space_size = params[kc.ACTION_SPACE_SIZE]
    
    # Generating agent data
    agents_data_df = generate_agents_data(num_agents, ratio_mutating, agent_attributes, simulation_timesteps, num_origins, num_destinations)
    agents = list() # Where we will store & return agents
    
    # Generating agent objects from generated agent data
    for _, row in agents_data_df.iterrows():
        row_dict = row.to_dict()

        id, start_time = row_dict[kc.AGENT_ID], row_dict[kc.AGENT_START_TIME]
        origin, destination = row_dict[kc.AGENT_ORIGIN], row_dict[kc.AGENT_DESTINATION]

        #if row_dict[kc.AGENT_KIND] == kc.TYPE_MACHINE: ##### Changed
        #    agent_params = params[kc.HUMAN_AGENT_PARAMETERS]
        #    initial_knowledge = free_flow_times[(origin, destination)]
        #    mutate_to = MachineAgent(id, start_time, origin, destination, params[kc.MACHINE_AGENT_PARAMETERS], action_space_size)
        #    new_agent = HumanAgent(id, start_time, origin, destination, agent_params, initial_knowledge, mutate_to)
        #    agents.append(new_agent)
        if row_dict[kc.AGENT_KIND] == kc.TYPE_HUMAN:
            agent_params = params[kc.HUMAN_AGENT_PARAMETERS]
            initial_knowledge = free_flow_times[(origin, destination)]
            human_noise = np.random.gumbel()
            agents.append(HumanAgent(id, start_time, origin, destination, agent_params, initial_knowledge, action_space_size, human_noise))
        else:
            print('[AGENT TYPE INVALID] Unrecognized agent type: ' + row_dict[kc.AGENT_KIND])

    print(f'[SUCCESS] Created agent objects (%d)' % (len(agents)))
    #print_agents(agents, agent_attributes, print_every=50)
    return agents



def generate_agents_data(num_agents, ratio_mutating, agent_attributes, simulation_timesteps, num_origins, num_destinations):

    """
    Generates agents data
    Constructs a dataframe, where each row is an agent and columns are attributes
    Saves it to specified named csv file
    """

    agents_df = pd.DataFrame(columns=agent_attributes)  # Where we store our agents
    i=0
    starttime=0

    for id in range(num_agents):
        # Generating agent data
        agent_type = kc.TYPE_HUMAN if (random.random() > ratio_mutating) else kc.TYPE_MACHINE ###### 80% of the agents are humans

        if id % 4 == 0:

            starttime += 2
        
        else:

            starttime = starttime

        if i % 4 == 0:  # O1, D1
            origin, destination = 0, 0
        elif i % 4 == 1:  # O1, D2
            origin, destination = 0, 1
        elif i % 4 == 2:  # O2, D1
            origin, destination = 1, 0
        elif i % 4 == 3:  # O2, D2
            origin, destination = 1, 1

        start_time = starttime
        
        i += 1  # Increment the counter for each agent


        #if i % 2 == 0:
        #    o=0
        #    d=0
        #    origin, destination = o,d
        #    start_time = starttime
        #    starttime +=2
        #    i+=1
        #elif i % 3 == 0:
        #    o=0
        #    d=1
        #    origin, destination = o,d
        #    start_time = starttime
        #    starttime +=2
        #    i+=1
        #elif i % 4 == 0:
        #    o=1
        #    d=0
        #    origin, destination = o,d
        #    start_time = starttime
        #    starttime +=2
        #    i+=1
        #elif i % 1 == 0:
        #    o=1
        #    d=1
        #    origin, destination = o,d
        #    start_time = starttime
        #    starttime +=2
        #    i+=1
        #else:

        #    raise ValueError('Not good')

        # Registering to the dataframe
        #print(id)
        #id = id + 3000
        agent_features = [id, origin, destination, start_time, agent_type]
        agent_dict = {attribute : feature for attribute, feature in zip(agent_attributes, agent_features)}
        agents_df.loc[id] = agent_dict

    save_to = make_dir(kc.RECORDS_FOLDER, kc.AGENTS_DATA_FILE_NAME)
    agents_df.to_csv(save_to, index = False)
    print('[SUCCESS] Generated agent data and saved to: ' + save_to)
    return agents_df



def print_agents(agents, agent_attributes, print_every=1):
    table = PrettyTable()
    table.field_names = agent_attributes

    for a in agents:
        if not (a.id % print_every):
            table.add_row([a.id, a.origin, a.destination, a.start_time, a.__class__.__name__])

    if print_every > 1: print("------ Showing every %dth agent ------" % (print_every))
    print(table)