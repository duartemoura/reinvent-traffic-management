from __future__ import absolute_import
from __future__ import print_function

import os
import boto3
from botocore.exceptions import NoCredentialsError
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

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

if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))
    upload_to_s3(os.path.join(plot_path, 'testing_settings.ini'), 'testing/testing_settings.ini')

    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue length (vehicles)')

    # Upload plots to S3
    plots = ['reward.png', 'queue.png']
    for plot in plots:
        local_plot_path = os.path.join(plot_path, f'plot_{plot.split(".")[0]}.png')
        s3_plot_path = f'testing/plots/{plot}'
        upload_to_s3(local_plot_path, s3_plot_path)
