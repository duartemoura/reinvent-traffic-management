import os
import sys
from shutil import copyfile
import boto3
from botocore.exceptions import ClientError
from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

def list_s3_models(bucket_name, prefix='models/'):
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        models = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.h5')]
        return models
    except Exception as e:
        print(f"Failed to list S3 bucket contents: {e}")
        return []

def download_model_from_s3(bucket_name, model_key, local_path):
    s3 = boto3.client('s3')
    try:
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
    # Load configuration
    config = import_test_configuration(config_file='testing_settings.ini')
    
    # Set up SUMO command
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    
    # Get S3 bucket name from environment variable
    s3_bucket = os.environ.get('S3_BUCKET')
    if not s3_bucket:
        print("S3_BUCKET environment variable is not set.")
        sys.exit(1)

    # Set up paths
    model_path = '/tmp/models'
    os.makedirs(model_path, exist_ok=True)
    plot_path = '/tmp/plots'
    os.makedirs(plot_path, exist_ok=True)

    # List available models in S3
    available_models = list_s3_models(s3_bucket)
    if not available_models:
        print("No models found in the S3 bucket. Please check your S3 bucket configuration.")
        sys.exit(1)
    
    print("Available models in S3:")
    for model in available_models:
        print(f"- {model}")

    # S3 model configuration
    model_key = f'trained_model/model_{config["model_to_test"]}.h5'
    local_model_path = os.path.join(model_path, os.path.basename(model_key))
    
    # Try to download the model from S3
    if not download_model_from_s3(s3_bucket, model_key, local_model_path):
        print("Falling back to local model...")
        local_model_path = f'models/model_{config["model_to_test"]}.h5'
        if not os.path.exists(local_model_path):
            print(f"Local model {local_model_path} not found. Please ensure a model file is available.")
            sys.exit(1)
    
    # Initialize model, traffic generator, and visualization
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
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))
    visualization.save_data_and_plot(data=simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    visualization.save_data_and_plot(data=simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue length (vehicles)')

if __name__ == "__main__":
    main()