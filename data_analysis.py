import csv
import numpy as np
import os
import pandas as pd

from pathlib import Path
from scipy.stats import entropy

class Table_record_creator:

    def __init__(self, number_of_episode, folder, day_limit, action_space):

        """

        Create one row for the final table\n

        Values:\n

        number_of_episode: Overall number of episode [int]\n
        folder: The name of the folder of the simulation [str]\n
        day_limit: The number  of day which should be examined [int]\n
        action_space: The number of the action space [int]


        """



        RoutingZoo = str(Path.home() / "Documents/Simulator_human_behaviour")
        self.route = RoutingZoo

        self.number_of_episode = number_of_episode +1
        self.folder = folder
        self.day_limit = day_limit
        self.action_space = action_space

        self.check_raw_folder()
        self.check_result_folder()

    
    def table_record(self):

        self.data_creator()
        data = self.table_record_creator()
    
        return data
    
    def data_creator(self):

        df = self.Raw_data_frame()
        df = df[df.day>self.day_limit]

        self.link_data()
        self.entropy_data(df)
        self.TT_data(df)

        print('Run was succesful')

    #Creating the raw data frame from the simulation values

    def Concater_to_raw_data(self):

        """

        Creating one joint dataframe from the records day by day
        
        """

        if not os.path.exists(f'{self.route}/raws/{self.folder}'):

            os.makedirs(f'{self.route}/raws/{self.folder}')
        
        for i in range(1,self.number_of_episode):

            #act = pd.read_csv(f'training_records/agents/{self.folder}/ep{i}.csv')
            #ep = pd.read_csv(f'training_records/episodes/{self.folder}/ep_ep{i}.csv')
            #df = pd.merge(act,ep,right_on='id',left_on='id')
            #cost = df.cost_table.str.split(pat=',', expand=True)

            #for name in range(self.action_space):

             #   cost = cost.rename(columns = {name:f'cost_{name}'})

            #df = df.merge(cost,right_index=True,left_index=True)
            #df = df.drop(columns = ['cost_table'])

            #utility = df.utility.str.split(pat=',', expand=True)

            #for name in range(self.action_space):

                #utility = utility.rename(columns = {name:f'U_{name}'})

            #df = df.merge(utility,right_index=True,left_index=True)
            #df = df.drop(columns = ['utility','kind'])
            #df['day'] = i
            act = pd.read_csv(f'{self.route}/training_records/agents/{self.folder}/ep{i}.csv')
            ep = pd.read_csv(f'{self.route}/training_records/episodes/{self.folder}/ep_ep{i}.csv')
            df = pd.merge(act,ep,right_on='id',left_on='id')
            cost = df.cost_table.str.split(pat=',', expand=True)

            for name in range(self.action_space):

                cost = cost.rename(columns = {name:f'cost_{name}'})

            df = df.merge(cost,right_index=True,left_index=True)
            df = df.drop(columns = ['cost_table'])

            utility = df.utilities.str.split(pat=',', expand=True)

            for name in range(self.action_space):

                utility = utility.rename(columns = {name:f'U_{name}'})
                utility[f'U_{name}'] = utility[f'U_{name}'].str.replace(']','')
                utility[f'U_{name}'] = utility[f'U_{name}'].str.replace('[','')


            noise = [list(map(float, row.strip('[]').split())) for row in df.noises]
            noise = pd.DataFrame(noise)

            for name in range(self.action_space):

                noise = noise.rename(columns = {name:f'noise_{name}'})

            df = df.merge(utility,right_index=True,left_index=True)
            df = df.merge(noise,right_index=True,left_index=True)
            df = df.drop(columns = ['utilities','noises','kind','reward_right','reward'])
            df['day'] = i
            df.to_csv(f'{self.route}/raws/{self.folder}/raw{i}.csv',index=False)
        
        det_main=pd.DataFrame([])

        for number in range(1,self.number_of_episode):

            det = pd.read_csv(f'{self.route}/training_records/detector/{self.folder}/detector_ep{number}.csv')
            det['day'] = number
            det_main = pd.concat([det_main,det])

        det_main = det_main.reset_index(drop=True)
        det_main.to_csv(f'{self.route}/training_records/detector/{self.folder}/all_det.csv',index=False)

    def Raw_data_frame(self,df_final = pd.DataFrame([])):
        """
        
        Create the joint raw data and a folder with all of the raw data
    
        """
        if not os.path.exists(f'{self.route}/raw_all'):
            os.makedirs(f'{self.route}/raw_all')


        self.Concater_to_raw_data()

        for i in range(1,self.number_of_episode):

            df = pd.read_csv(f'{self.route}/raws/{self.folder}/raw{i}.csv')
            df_ = pd.concat([df_final,df])
            df_final = df_


        df_final['value'] = df_final.apply(lambda row: row[f'cost_{int(row["action"])}'], axis=1)
        df_final['label'] = df_final.apply(lambda row: f'cost_{int(row["action"])}', axis=1)
        
        df_final.to_csv(f'{self.route}/raw_all/raw_all_{self.folder}.csv',index=False)

        return df_final
    
    #Creating the records for the table record
    
    def link_data(self):

        detector = self.detector_std(pd.read_csv(f'{self.route}/training_records/detector/{self.folder}/all_det.csv'))

        with open(f'{self.route}/result/{self.folder}_link.csv', 'w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(['name', 'link'])

            for key, value in detector.items():

                writer.writerow([key, value])
            
        result_link = {}

        df = pd.read_csv(f'{self.route}/result/{self.folder}_link.csv')
        result_link[self.folder] = df.values[2, 1]

        with open(f'{self.route}/result/link_result.csv', 'w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(['name', 'value'])

            for key, value in result_link.items():

                writer.writerow([key, value])

        os.remove(f'{self.route}/result/{self.folder}_link.csv')

    def entropy_data(self, df):

        entropy = self.entropy_calculator(df)

        with open(f'{self.route}/result/{self.folder}_entropy.csv', 'w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(['name', 'entropy'])

            for key, value in entropy.items():

                writer.writerow([key, value])

        result_entropy = {}

        df = pd.read_csv(f'{self.route}/result/{self.folder}_entropy.csv')

        result_entropy[self.folder] = [np.mean(df.values[:, 1]),np.std(df.values[:, 1])]

        with open(f'{self.route}/result/entropy_result.csv', 'w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(['name', 'value'])

            for key, value in result_entropy.items():

                writer.writerow([key, value])
        
        os.remove(f'{self.route}/result/{self.folder}_entropy.csv')

    def TT_data(self, df):

        TravelTime = self.TotalTravelTime(df=df)

        with open(f'{self.route}/result/{self.folder}_TT.csv', 'w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(['day', 'TT'])

            for key, value in TravelTime.items():

                writer.writerow([key, value])

        result_TT = {}

        df = pd.read_csv(f'{self.route}/result/{self.folder}_TT.csv')

        result_TT[self.folder] = [np.mean(df.values[:, 1]),np.std(df.values[:, 1])]


        with open(f'{self.route}/result/TT_result.csv', 'w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(['name', 'value'])

            for key, value in result_TT.items():

                writer.writerow([key, value])
        
        os.remove(f'{self.route}/result/{self.folder}_TT.csv')

    #Creating the final data as a table record

    def table_record_creator(self):

        df = pd.read_csv(f'{self.route}/result/entropy_result.csv')
        df.value = df.value.str.replace('[',"")
        df.value = df.value.str.replace(']',"")
        entropy = df.name.str.split('_',expand=True)
        entropy = entropy.rename(columns={0:'network',1:'model',2:'demand',3:'Bounded',4:'Greedy'})
        entropy1 = df.value.str.split(',',expand=True)
        entropy1 = entropy1.rename(columns={0:'entropy_value',1:'entropy_std'})
        entropy = entropy.merge(entropy1,right_index=True,left_index=True)

        df = pd.read_csv(f'{self.route}/result/link_result.csv')
        df.value = df.value.str.replace('[',"")
        df.value = df.value.str.replace("'","")
        df.value = df.value.str.replace(']',"")
        link = df.name.str.split('_',expand=True)
        link = link.rename(columns={0:'network',1:'model',2:'demand',3:'Bounded',4:'Greedy'})
        link1 = df.value.str.split(',',expand=True)
        link1 = link1.rename(columns={0:'link_value',1:'link_std'})
        link = link.merge(link1,right_index=True,left_index=True)

        df = pd.read_csv(f'{self.route}/result/TT_result.csv')
        df.name = df.name.str.replace('_TT.csv',"")
        df.value = df.value.str.replace('[',"")
        df.value = df.value.str.replace(']',"")
        TT = df.name.str.split('_',expand=True)
        TT = TT.rename(columns={0:'network',1:'model',2:'demand',3:'Bounded',4:'Greedy'})
        TT1 = df.value.str.split(',',expand=True)
        TT1 = TT1.rename(columns={0:'TT_value',1:'TT_std'})
        TT = TT.merge(TT1,right_index=True,left_index=True)

        data = link.merge(TT,right_on=['network','model','demand','Bounded','Greedy'],left_on=['network','model','demand','Bounded','Greedy'])
        data = data.merge(entropy,right_on=['network','model','demand','Bounded','Greedy'],left_on=['network','model','demand','Bounded','Greedy'])

        data.link_value = data.link_value.astype('float').round(0)
        data.entropy_value = data.entropy_value.astype('float').round(2)
        data.TT_value = data.TT_value.astype('float').round(0)
        data.link_std = data.link_std.astype('float').round(2)
        data.entropy_std = data.entropy_std.astype('float').round(2)
        data.TT_std = data.TT_std.astype('float').round(2)

        return data
    
    #Utilities

    def detector_std(self, df):

        detector = df

        name = detector.detid.unique()

        detector_std = {}

        for n in name:

            detector_std[n] = [np.mean(detector[(detector.detid==n)].flow),np.std(detector[(detector.detid==n)].flow)]
        
        return detector_std
    

    def entropy_calculator(self, df):

        df_sequence = df
        df_calc = df_sequence
        df_calc = df_calc.groupby(['origin','destination','id'])['label'].value_counts().unstack(fill_value=0).reset_index()
        df_calc = df_calc.sort_values('id')
        keys_to_check = ['cost_1','cost_2', 'cost_3']

        existing_keys = [key for key in keys_to_check if key in df_calc.columns]

        counts = df_calc[existing_keys].values

        entropy_dict = {}

        for i in range(len(counts)):

            calc = counts[i]
            total_count = sum(calc)
            probabilities = [i / total_count for i in calc]
            entropy_value = entropy(probabilities,base=2)

            if np.isnan(entropy_value):

                entropy_value = 1

            entropy_dict[i] = entropy_value
        
        return entropy_dict
    
    def TotalTravelTime(self,df):

        days = df.day.unique()

        TT = {} 

        for d in days:

            value = df[df.day==d].sum().travel_time
            TT[d] = value
        
        return TT
    
    def check_result_folder(self):
    
        if not os.path.exists(f'{self.route}/result'):

            os.makedirs(f'{self.route}/result')

    def check_raw_folder(self):
    
        if not os.path.exists(f'{self.route}/raws'):

            os.makedirs(f'{self.route}/raws')