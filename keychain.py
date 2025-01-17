class Keychain:

    """
    This is where we store our file paths, parameter access keys and other constants reliably
    When change needed, just fix it here! Avoid hardcoding...
    """

    ################## RELATIVE FILE PATHS ####################

    PARAMS_PATH = "params.json"


    RECORDS_FOLDER = "training_records"
    EPISODES_LOGS_FOLDER = "episodes"
    DETECTOR_LOGS_FOLDER = 'detector'
    AGENTS_LOGS_FOLDER = "agents"

    AGENTS_DATA_FILE_NAME = "agents_data.csv"
    SIMULATION_LENGTH_LOG_FILE_NAME = "simulation_length.txt"
    PATHS_CSV_FILE_NAME = "paths.csv"
    FREE_FLOW_TIMES_CSV_FILE_NAME = "free_flow_times.csv"

    PLOTS_FOLDER = "plots"
    REWARDS_PLOT_FILE_NAME = "rewards.png"
    REWARDS_DIST_PLOT_FILE_NAME = "rewards_dist.png"
    FF_TRAVEL_TIME_PLOT_FILE_NAME = "ff_travel_time.png"
    FLOWS_PLOT_FILE_NAME = "flows.png"
    SIMULATION_LENGTH_PLOT_FILE_NAME = "simulation_length.png"
    ACTIONS_PLOT_FILE_NAME = "actions.png"
    ACTIONS_SHIFTS_PLOT_FILE_NAME = "actions_shifts.png"
    MACHINE_AGENTS_EPSILONS_PLOT_FILE_NAME = "epsilons.png"
    
    ###########################################################
    
    

    
    ################ PARAMETER ACCESS KEYS ####################

    AGENTS_GENERATION_PARAMETERS = "agent_generation_parameters"
    TRAINING_PARAMETERS = "training_parameters"
    ENVIRONMENT_PARAMETERS = "environment_parameters"
    SIMULATION_PARAMETERS = "simulation_parameters"

    # Agent generation
    AGENTS_DATA_PATH = "agents_data_path"
    NUM_AGENTS = "num_agents"
    RATIO_MUTATING = "ratio_mutating"
    SIMULATION_TIMESTEPS = "simulation_timesteps"

    ACTION_SPACE_SIZE = "action_space_size"
    MACHINE_AGENT_PARAMETERS = "machine_agent_parameters"
    HUMAN_AGENT_PARAMETERS = "human_agent_parameters"

    MIN_ALPHA = "min_alpha"
    MAX_ALPHA = "max_alpha"
    MIN_EPSILON = "min_epsilon"
    MAX_EPSILON = "max_epsilon"
    MIN_EPS_DECAY = "min_eps_decay"
    MAX_EPS_DECAY = "max_eps_decay"
    GAMMA = "gamma"

    ALPHA = "alpha"
    BETA = "beta"
    BETA_RANDOMNESS = "beta_randomness"
    REMEMBER = 'remember'
    ALPHA_NULLA = 'alpha_nulla'
    ALPHA_J = 'alpha_j'
    DELTA = 'delta'
    GREEDY = 'greedy'
    LEARNING_TYPE = 'learning_type'
    ACT_TYPE = 'act_type'
    MU = 'mu'
    NOISE_ALPHA = 'noise_alpha'
    RANDOMNES = 'randomness'


    # Training
    NUM_EPISODES = "num_episodes"
    REMEMBER_EVERY = "remember_every"
    MUTATION_TIME = "mutation_time"

    # Environment
    TRANSPORT_PENALTY = "transport_penalty"

    # Simulation
    SUMO_TYPE = "sumo_type"
    SEED = 'seed'
    SUMO_CONFIG_PATH = "sumo_config_path"
    CONNECTION_FILE_PATH = "connection_file_path"
    EDGE_FILE_PATH = "edge_file_path" 
    ROUTE_FILE_PATH = "route_file_path"
    ROUTES_XML_SAVE_PATH = "routes_xml_save_path"
    DETECTOR_XML_SAVE_PATH = "detector_xml_save_path"
    PATHS_SAVE_PATH = "paths_save_path"
    NUMBER_OF_PATHS = "number_of_paths"
    ROUTE_UTILITY_COEFFS = "route_utility_coeffs"
    ORIGINS = "origins"
    DESTINATIONS = "destinations"
    WEIGHT = "weight"
    NUM_SAMPLES = "num_samples"
    MAX_PATH_LENGTH = "max_path_length"
    PATHS_CSV_SAVE_PATH = "path_csv_save_path"
    PATHS_CSV_SAVE_DETECTORS = "path_csv_save_detectors"

    # Recorder
    RECORDER_PARAMETERS = "recorder_parameters"

    # Plotter
    PLOTTER_PARAMETERS = "plotter_parameters"

    ###########################################################
    
    

    
    ####################### ELSE ##############################

    SMALL_BUT_NOT_ZERO = 1e-14
    NOT_AVAILABLE = "N/A"

    SUMO_HOME = "SUMO_HOME"

    AGENT_ATTRIBUTES = "agent_attributes"
    
    # Common dataframe column headers
    AGENT_ID = "id"
    AGENT_KIND = "kind"
    AGENT_ORIGIN = "origin"
    AGENT_DESTINATION = "destination"
    AGENT_START_TIME = "start_time"
    TO_MUTATE = "to_mutate"
    ACTION = "action"
    Utility = 'utility'
    Noise = 'noise'
    SUMO_ACTION = "sumo_action"
    REWARD = "reward"
    COST = "cost"
    Q_TABLE = "q_table"
    EPSILON = "epsilon"
    EPSILON_DECAY_RATE = "epsilon_decay_rate"
    ARRIVAL_TIME = "arrival_time"
    COST_TABLE = "cost_table"
    TRAVEL_TIME = "travel_time"
    HUMANS = "humans"
    MACHINES = "machines"
    ALL = "all"
    PATH_INDEX = "path_index"
    FREE_FLOW_TIME = "free_flow_time"
    ORIGIN = "origin"
    DESTINATION = "destination"
    PATH = "path"
    

    # Agent type encodings
    TYPE_HUMAN = "h"
    TYPE_MACHINE = "m"

    ###########################################################