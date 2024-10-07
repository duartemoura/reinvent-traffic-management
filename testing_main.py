import os
import sys
from shutil import copyfile
import boto3
from botocore.exceptions import ClientError
from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo

def create_model_paths(models_path_name, model_number):
    """Create necessary directories for the model"""
    base_path = os.getcwd()
    model_folder_path = os.path.join(base_path, models_path_name, f'model_{model_number}')
    plot_path = os.path.join(model_folder_path, 'test')
    
    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    
    return model_folder_path, plot_path

def download_model_from_s3(bucket_name, model_key, local_path):
    s3 = boto3.client('s3')
    try:
        print(f"Attempting to download {model_key} from {bucket_name} to {local_path}")
        s3.download_file(bucket_name, model_key, local_path)
        print(f"Model {model_key} downloaded successfully from S3 to {local_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"The model file {model_key} does not exist in the S3 bucket.")
        else:
            print(f"Failed to download model from S3: {e}")
        return False

def main():
    print("Starting main function")
    
    # Load configuration
    config = import_test_configuration(config_file='testing_settings.ini')
    print(f"Loaded configuration: {config}")
    
    # Set up SUMO command
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    print(f"SUMO command: {sumo_cmd}")
    
    # Set up paths
    model_folder_path, plot_path = create_model_paths(config['models_path_name'], config['model_to_test'])
    print(f"Model folder path: {model_folder_path}")
    print(f"Plot path: {plot_path}")

    # S3 model configuration
    s3_bucket = config['s3']['bucket_name']
    model_key = config['s3']['model_key']
    local_model_path = os.path.join(model_folder_path, os.path.basename(model_key))
    print(f"S3 bucket: {s3_bucket}")
    print(f"Model key: {model_key}")
    print(f"Local model path: {local_model_path}")
    
    # Try to download the model from S3
    if not download_model_from_s3(s3_bucket, model_key, local_model_path):
        print("Falling back to local model...")
        local_model_path = os.path.join(model_folder_path, f'model_{config["model_to_test"]}.h5')
        if not os.path.exists(local_model_path):
            print(f"Local model {local_model_path} not found. Please ensure a model file is available.")
            sys.exit(1)
    
    # Initialize model, traffic generator, and visualization
    print("Initializing model, traffic generator, and visualization")
    model = TestModel(
        input_dim=config['num_states'],
        model_path=local_model_path
    )
    traffic_gen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )
    visualization = Visualization(
        plot_path, 
        dpi=96
    )
    
    # Set up and run simulation
    print("Setting up simulation")
    simulation = Simulation(
        model,
        traffic_gen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )
    
    print('\n----- Test episode')
    simulation_time = simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')
    print("----- Testing info saved at:", plot_path)
    
    # Save results
    print("Saving results")
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))
    visualization.save_data_and_plot(data=simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    visualization.save_data_and_plot(data=simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue length (vehicles)')

if __name__ == "__main__":
    print("Starting testing process...")
    main()