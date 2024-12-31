import os
import pandas as pd
import shutil

class Utilities():

    def __init__(self, env, RoutingZoo, episode, **kwargs):

        self.env = env
        self.RoutingZoo = RoutingZoo
        self.episode = episode
        self.model = self.name(**kwargs)

        self.check_model_folder()

    def data(self):

        id = []
        Utilities = []
        Noises = []


        for a in range(len(self.env.human_agents)):

            id.append(self.env.human_agents[a].id)
            Utilities.append(self.env.human_agents[a].stored_utilities)
            Noises.append(self.env.human_agents[a].stored_noises)

            data = pd.DataFrame([id,Utilities,Noises]).T
            data = data.rename(columns={0:'id',1:'utilities',2:'noises'})
            data.to_csv(f'{self.RoutingZoo}/training_records/agents/{self.model}/ep{self.episode+1}.csv',index=False)

    
    def Replace_data(self):

        os.rename(f'/Users/zoltanvarga/Documents/RouteRL/tutorials/RoutingZoo/training_records/episodes/ep{self.episode+1}.csv',f'/Users/zoltanvarga/Documents/RouteRL/tutorials/RoutingZoo/training_records/episodes/ep_ep{self.episode+1}.csv')

        source_episode = f'/Users/zoltanvarga/Documents/RouteRL/tutorials/RoutingZoo/training_records/episodes/ep_ep{self.episode+1}.csv'
        destination_episode = f'{self.RoutingZoo}/training_records/episodes/{self.model}/ep_ep{self.episode+1}.csv'

        shutil.copy2(source_episode, destination_episode)

        source_det = f'/Users/zoltanvarga/Documents/RouteRL/tutorials/RoutingZoo/training_records/detector/detector_ep{self.episode+1}.csv'
        destination_det = f'{self.RoutingZoo}/training_records/detector/{self.model}/detector_ep{self.episode+1}.csv'

        shutil.copy2(source_det, destination_det)

    def check_model_folder(self):
    
        if not os.path.exists(f'{self.RoutingZoo}/training_records/agents/{self.model}'):

            os.makedirs(f'{self.RoutingZoo}/training_records/agents/{self.model}')

        if not os.path.exists(f'{self.RoutingZoo}/training_records/episodes/{self.model}'):

            os.makedirs(f'{self.RoutingZoo}/training_records/episodes/{self.model}')

        if not os.path.exists(f'{self.RoutingZoo}/training_records/detector/{self.model}'):

            os.makedirs(f'{self.RoutingZoo}/training_records/detector/{self.model}')
            

    def name(self, **kwargs):

        network = kwargs['network']
        model = kwargs['learning_type']
        demand = kwargs['demand']
        Bounded = kwargs['gamma_c']
        Greedy = kwargs['greedy']

        return f'{network}_{model}_{demand}_{Bounded}_{Greedy}'
