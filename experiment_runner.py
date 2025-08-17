import rospy
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import time
import json
from typing import List, Dict, Tuple
from experiment_runner import ExperimentRunner
from utils import setup_logging, create_experiment_dir
from agents import Agent

# Set up logging
logger = setup_logging('experiment_runner')

class ExperimentRunner:
    """
    ExperimentRunner class to automate the execution of experiments, collect metrics,
    and generate reports for the Saliency-Biased Frontier Exploration project.

    ...

    Attributes
    ----------
    experiment_dir : str
        Directory path to store experiment results and logs.
    config : dict
        Experiment configuration loaded from a YAML file.
    agents : list of Agent
        List of agent objects used in the experiments.
    envs : list of str
        List of environment names to run experiments in.
    metrics : dict
        Dictionary to store collected metrics.
    results : dict
        Dictionary to store experiment results.

    Methods
    -------
    run_experiment(self, agent: Agent, env: str)
        Run an experiment with a given agent in a specific environment.
    collect_metrics(self, agent: Agent, env: str)
        Collect and compute metrics for a given experiment.
    save_results(self)
        Save experiment results and metrics to disk.
    generate_plots(self)
        Generate plots for the experiment results.
    create_summary_report(self)
        Create a summary report of the experiment results.
    parameter_sweep(self, param: str, values: List[float or int])
        Perform a parameter sweep over a range of values for a given parameter.

    """

    def __init__(self, config_file: str):
        """
        Initialize the ExperimentRunner with the experiment configuration.

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file.

        """
        self.experiment_dir = create_experiment_dir()
        self.config = self._load_config(config_file)
        self.agents = []
        self.envs = self.config['environments']
        self.metrics = {}
        self.results = {}

        # Initialize agents
        for agent_config in self.config['agents']:
            agent = Agent(**agent_config)
            self.agents.append(agent)

    def run_experiment(self, agent: Agent, env: str) -> None:
        """
        Run an experiment with a given agent in a specific environment.

        Parameters
        ----------
        agent : Agent
            Agent object to use in the experiment.
        env : str
            Name of the environment to run the experiment in.

        Returns
        -------
        None

        """
        try:
            # Set up experiment
            rospy.init_node('experiment_runner', anonymous=True)
            agent.setup(env)

            # Run the experiment
            logger.info(f"Running experiment with agent '{agent.name}' in environment '{env}'")
            agent.explore()
            agent.teardown()

            # Collect metrics
            self.collect_metrics(agent, env)

        except Exception as e:
            logger.error(f"Experiment with agent '{agent.name}' in environment '{env}' failed: {e}")
            raise e

    def collect_metrics(self, agent: Agent, env: str) -> None:
        """
        Collect and compute metrics for a given experiment.

        Parameters
        ----------
        agent : Agent
            Agent object used in the experiment.
        env : str
            Name of the environment the experiment was run in.

        Returns
        -------
        None

        """
        try:
            # Get environment-specific metrics
            env_metrics = agent.get_environment_metrics(env)

            # Compute exploration metrics
            exploration_metrics = self._compute_exploration_metrics(agent, env)

            # Compute saliency metrics
            saliency_metrics = self._compute_saliency_metrics(agent, env)

            # Store metrics
            self.metrics[f'{agent.name}_{env}'] = {
                'environment': env_metrics,
                'exploration': exploration_metrics,
                'saliency': saliency_metrics
            }

        except Exception as e:
            logger.error(f"Metric collection for agent '{agent.name}' in environment '{env}' failed: {e}")
            raise e

    def save_results(self) -> None:
        """
        Save experiment results and metrics to disk.

        Returns
        -------
        None

        """
        try:
            results_file = os.path.join(self.experiment_dir, 'results.csv')
            metrics_file = os.path.join(self.experiment_dir, 'metrics.json')

            # Save results as CSV
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(results_file, index=False)
            logger.info(f"Saved experiment results to {results_file}")

            # Save metrics as JSON
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            logger.info(f"Saved experiment metrics to {metrics_file}")

        except Exception as e:
            logger.error(f"Saving experiment results and metrics failed: {e}")
            raise e

    def generate_plots(self) -> None:
        """
        Generate plots for the experiment results.

        Returns
        -------
        None

        """
        try:
            # Create directory for plots
            plots_dir = os.path.join(self.experiment_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)

            # Plot exploration metrics
            for agent, agent_results in self.results.items():
                for env, env_results in agent_results.items():
                    # Plot exploration curve
                    exploration_curve = env_results['exploration_curve']
                    plt.figure()
                    sns.lineplot(x='time', y='explored_area', data=exploration_curve)
                    plt.title(f"Exploration Curve for Agent {agent} in Environment {env}")
                    plt.xlabel('Time (s)')
                    plt.ylabel('Explored Area (m2)')
                    plt.tight_layout()
                    plot_file = os.path.join(plots_dir, f'exploration_curve_{agent}_{env}.png')
                    plt.savefig(plot_file)
                    logger.info(f"Saved exploration curve plot to {plot_file}")

                    # Plot other metrics...

            # Plot saliency maps...

        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            raise e

    def create_summary_report(self) -> None:
        """
        Create a summary report of the experiment results.

        Returns
        -------
        None

        """
        try:
            # Create directory for reports
            reports_dir = os.path.join(self.experiment_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)

            # Generate summary report
            report_file = os.path.join(reports_dir, 'summary_report.md')
            with open(report_file, 'w') as f:
                f.write("# Experiment Summary Report\n")
                f.write("## Agent Performance Comparison\n")
                # Insert table comparing agent performance here
                # ...
                f.write("## Environment Comparison\n")
                # Insert table comparing environment performance here
                # ...

            logger.info(f"Created summary report at {report_file}")

        except Exception as e:
            logger.error(f"Summary report creation failed: {e}")
            raise e

    def parameter_sweep(self, param: str, values: List[float or int]) -> None:
        """
        Perform a parameter sweep over a range of values for a given parameter.

        Parameters
        ----------
        param : str
            Name of the parameter to sweep.
        values : list of float or int
            List of values to sweep the parameter over.

        Returns
        -------
        None

        """
        try:
            # Create directory for parameter sweeps
            sweeps_dir = os.path.join(self.experiment_dir, 'parameter_sweeps')
            os.makedirs(sweeps_dir, exist_ok=True)

            # Perform parameter sweep
            for value in values:
                # Update parameter value in agent configuration
                # ...

                # Run experiments for each environment
                for env in self.envs:
                    self.run_experiment(self.agents[0], env)  # Assuming single agent for simplicity

                # Save results and generate plots for this parameter value
                self.save_results()
                self.generate_plots()

                # Reset agent configuration
                # ...

        except Exception as e:
            logger.error(f"Parameter sweep for '{param}' failed: {e}")
            raise e

    def _load_config(self, config_file: str) -> Dict:
        """
        Load experiment configuration from a YAML file.

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file.

        Returns
        -------
        dict
            Loaded configuration.

        """
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config

        except FileNotFoundError:
            logger.error(f"Configuration file '{config_file}' not found.")
            raise

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file '{config_file}': {e}")
            raise

    def _compute_exploration_metrics(self, agent: Agent, env: str) -> Dict:
        """
        Compute exploration metrics for a given experiment.

        Parameters
        ----------
        agent : Agent
            Agent object used in the experiment.
        env : str
            Name of the environment the experiment was run in.

        Returns
        -------
        dict
            Dictionary of computed exploration metrics.

        """
        # Get exploration curve data
        exploration_curve = agent.get_exploration_curve(env)

        # Compute additional metrics
        total_time = exploration_curve['time'][-1]
        max_velocity = np.max(agent.velocity_history)
        # ...

        return {
            'total_time': total_time,
            'max_velocity': max_velocity,
            # ...
        }

    def _compute_saliency_metrics(self, agent: Agent, env: str) -> Dict:
        """
        Compute saliency metrics for a given experiment.

        Parameters
        ----------
        agent : Agent
            Agent object used in the experiment.
        env : str
            Name of the environment the experiment was run in.

        Returns
        -------
        dict
            Dictionary of computed saliency metrics.

        """
        # Get saliency map data
        saliency_map = agent.get_saliency_map(env)

        # Compute saliency metrics
        num_salient_regions = len(np.unique(saliency_map)) - 1
        avg_saliency_value = np.mean(saliency_map[saliency_map > 0])
        # ...

        return {
            'num_salient_regions': num_salient_regions,
            'avg_saliency_value': avg_saliency_value,
            # ...
        }

def main():
    # Example usage of ExperimentRunner class
    config_file = 'experiment_config.yaml'
    runner = ExperimentRunner(config_file)

    # Run experiments
    for agent in runner.agents:
        for env in runner.envs:
            runner.run_experiment(agent, env)

    # Collect and save results
    runner.save_results()

    # Generate plots and summary report
    runner.generate_plots()
    runner.create_summary_report()

    # Perform parameter sweep (optional)
    param_to_sweep = 'exploration_range'
    values_to_sweep = [10, 20, 30]
    runner.parameter_sweep(param_to_sweep, values_to_sweep)

if __name__ == '__main__':
    main()