from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
import boto3
from botocore.exceptions import NoCredentialsError
from shutil import copyfile

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path

# Initialize S3 client
s3 = boto3.client('s3')
S3_BUCKET = os.environ.get('S3_BUCKET')  # Ensure this environment variable is set in SageMaker

def upload_to_s3(local_path, s3_path):
    try:
        s3.upload_file(local_path, S3_BUCKET, s3_path)
        print(f"Uploaded {local_path} to s3://{S3_BUCKET}/{s3_path}")
    except FileNotFoundError:
        print(f"The file {local_path} was not found")
    except NoCredentialsError:
        print("Credentials not available")

def set_train_path(models_path_name):
    # Create the main directory if it doesn't exist
    if not os.path.exists(models_path_name):
        os.makedirs(models_path_name)
        print(f"Created main directory: {models_path_name}")

    # Get the list of directories or files in models_path_name
    dir_content = os.listdir(models_path_name)

    # Filter to get previous version numbers (assuming your versions are named like 'model_1', 'model_2', ...)
    previous_versions = [
        int(name.split("_")[1]) 
        for name in dir_content 
        if len(name.split("_")) > 1 and name.split("_")[1].isdigit()
    ]

    # Calculate the next version number
    next_version = max(previous_versions, default=0) + 1

    # Create a new path for the model within the main folder
    new_model_path = os.path.join(models_path_name, f'model_{next_version}')
    os.makedirs(new_model_path)
    print(f"Created model directory: {new_model_path}")

    return new_model_path

if __name__ == "__main__":
    # Set the models path to the directory where your Docker files are located
    models_path_name = 'models'  # Change this to the desired folder name
    path = set_train_path(models_path_name)

    # Load training configuration
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    # Initialize models and training components
    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time + training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    # Save model after the simulation
    model_local_path = os.path.join(path, 'trained_model.h5')
    Model.save_model(path)  # Save model artifacts in the specified path

    # Check if the model file was created successfully
    if os.path.isfile(model_local_path):
        upload_to_s3(model_local_path, f'models/model_{os.path.basename(model_local_path)}')  # Use a clear path for the model
    else:
        print(f"Model file not found at {model_local_path}. Ensure it was saved correctly.")

    # Upload the training settings file
    training_settings_path = os.path.join(path, 'training_settings.ini')
    copyfile(src='training_settings.ini', dst=training_settings_path)
    upload_to_s3(training_settings_path, 'models/training_settings.ini')

    # Save visualization data
    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')

    # Upload plots to S3
    plots = ['reward.png', 'delay.png', 'queue.png']
    for plot in plots:
        local_plot_path = os.path.join(path, f'plot_{plot.split(".")[0]}.png')
        s3_plot_path = f'plots/{plot}'
        upload_to_s3(local_plot_path, s3_plot_path)
