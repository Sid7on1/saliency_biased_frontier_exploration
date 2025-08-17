import numpy as np
import geometry_msgs.msg as geom_msgs
from frontier_detection import detect_frontiers
from saliency_computation import compute_saliency_map
from typing import List, Tuple, Callable
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
VELOCITY_THRESHOLD = 0.1  # m/s
MAX_DISTANCE = 10.0  # m
MAP_RESOLUTION = 0.1  # m

# Custom exception classes
class InvalidFrontierException(Exception):
    pass

class ExplorationStrategy:
    """
    Base class for exploration strategies
    """
    def __init__(self, map_size: Tuple[int, int], resolution: float):
        self.map_size = map_size
        self.resolution = resolution
        self.map_scale = np.array([map_size[0] * resolution, map_size[1] * resolution])
        self.map_center = self.map_scale / 2.0
        self.occupied_map = None
        self.free_map = None
        self.exploration_map = None
        self.saliency_map = None

    def update_maps(self, occupied_map: np.ndarray, free_map: np.ndarray, saliency_map: np.ndarray) -> None:
        """
        Update the local maps used for exploration strategy computation.

        :param occupied_map: 2D numpy array representing the occupied space.
        :param free_map: 2D numpy array representing the free space.
        :param saliency_map: 2D numpy array representing the saliency values.
        """
        self.occupied_map = occupied_map
        self.free_map = free_map
        self.exploration_map = free_map.copy()
        self.saliency_map = saliency_map

    def compute_utility(self, position: np.ndarray, frontier: np.ndarray) -> float:
        """
        Compute the utility value for a given position and frontier.

        :param position: Current position of the robot as a 2D numpy array.
        :param frontier: 2D numpy array representing the frontier.
        :return: Utility value as a float.
        """
        raise NotImplementedError

    def apply_saliency_bias(self, utility_values: np.ndarray, saliency_values: np.ndarray) -> np.ndarray:
        """
        Apply saliency bias to the utility values.

        :param utility_values: 2D numpy array of utility values.
        :param saliency_values: 2D numpy array of saliency values.
        :return: Modified utility values with saliency bias applied.
        """
        # Normalize saliency values
        normalized_saliency = (saliency_values - saliency_values.min()) / (saliency_values.max() - saliency_values.min() + 1e-6)

        # Apply saliency bias to utility values
        biased_utility = utility_values * normalized_saliency

        return biased_utility

    def select_best_frontier(self, utility_values: np.ndarray) -> np.ndarray:
        """
        Select the best frontier based on the utility values.

        :param utility_values: 2D numpy array of utility values.
        :return: 2D numpy array representing the selected frontier.
        """
        # Find the frontier with the highest utility value
        max_utility = utility_values.max()
        max_utility_coords = np.where(utility_values == max_utility)

        # Check if a valid frontier was found
        if len(max_utility_coords[0]) == 0:
            raise InvalidFrontierException("No valid frontier found.")

        # Convert coordinates to a numpy array
        max_utility_coord = np.array([max_utility_coords[1][0], max_utility_coords[0][0]])

        # Convert to world coordinates
        world_coord = max_utility_coord * self.resolution + self.map_center

        return world_coord

    def normalize_values(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize the input values to the range [0, 1].

        :param values: Input values as a numpy array.
        :return: Normalized values as a numpy array.
        """
        return (values - values.min()) / (values.max() - values.min() + 1e-6)

class NearestFrontierStrategy(ExplorationStrategy):
    """
    Exploration strategy that selects the nearest frontier.
    """
    def __init__(self, map_size: Tuple[int, int], resolution: float):
        super().__init__(map_size, resolution)

    def compute_utility(self, position: np.ndarray, frontier: np.ndarray) -> float:
        """
        Compute the utility value for the nearest frontier strategy.

        :param position: Current position of the robot as a 2D numpy array.
        :param frontier: 2D numpy array representing the frontier.
        :return: Utility value as a float.
        """
        # Calculate distance to the frontier
        distance_to_frontier = np.linalg.norm(frontier - position)

        # Convert distance to utility value
        utility_value = 1.0 / (1.0 + distance_to_frontier)

        return utility_value

class InformationGainStrategy(ExplorationStrategy):
    """
    Exploration strategy that maximizes information gain.
    """
    def __init__(self, map_size: Tuple[int, int], resolution: float):
        super().__init__(map_size, resolution)

    def compute_information_gain(self, position: np.ndarray) -> float:
        """
        Compute the information gain for the current position.

        :param position: Current position of the robot as a 2D numpy array.
        :return: Information gain value as a float.
        """
        # Calculate the number of unknown cells
        unknown_cells = np.sum(self.exploration_map == 0)

        # Convert position to cell coordinates
        cell_pos = (position / self.resolution).astype(int)

        # Define a 3x3 kernel for local information gain calculation
        kernel = np.ones((3, 3))

        # Calculate local information gain
        local_info_gain = np.sum(self.exploration_map[max(0, cell_pos[0]-1):min(self.map_size[0], cell_pos[0]+2),
                                                 max(0, cell_pos[1]-1):min(self.map_size[1], cell_pos[1]+2)] * kernel)

        # Calculate information gain ratio
        info_gain_ratio = local_info_gain / unknown_cells

        return info_gain_ratio

    def compute_utility(self, position: np.ndarray, frontier: np.ndarray) -> float:
        """
        Compute the utility value for the information gain strategy.

        :param position: Current position of the robot as a 2D numpy array.
        :param frontier: 2D numpy array representing the frontier.
        :return: Utility value as a float.
        """
        # Calculate distance to the frontier
        distance_to_frontier = np.linalg.norm(frontier - position)

        # Calculate information gain for the current position
        info_gain = self.compute_information_gain(position)

        # Convert distance and information gain to utility value
        utility_value = info_gain / (1.0 + distance_to_frontier)

        return utility_value

class PerfectInformationGainStrategy(ExplorationStrategy):
    """
    Exploration strategy that maximizes perfect information gain.
    """
    def __init__(self, map_size: Tuple[int, int], resolution: float):
        super().__init__(map_size, resolution)

    def compute_perfect_information_gain(self, position: np.ndarray, frontier: np.ndarray) -> float:
        """
        Compute the perfect information gain for a given position and frontier.

        :param position: Current position of the robot as a 2D numpy array.
        :param frontier: 2D numpy array representing the frontier.
        :return: Perfect information gain value as a float.
        """
        # Calculate distance to the frontier
        distance_to_frontier = np.linalg.norm(frontier - position)

        # Convert position to cell coordinates
        cell_pos = (position / self.resolution).astype(int)

        # Define a 3x3 kernel for local information gain calculation
        kernel = np.ones((3, 3))

        # Calculate local information gain at the frontier
        local_frontier_info = np.sum(self.exploration_map[max(0, frontier[0]-1):min(self.map_size[0], frontier[0]+2),
                                                     max(0, frontier[1]-1):min(self.map_size[1], frontier[1]+2)] * kernel)

        # Calculate local information gain at the current position
        local_pos_info = np.sum(self.exploration_map[max(0, cell_pos[0]-1):min(self.map_size[0], cell_pos[0]+2),
                                                  max(0, cell_pos[1]-1):min(self.map_size[1], cell_pos[1]+2)] * kernel)

        # Calculate perfect information gain ratio
        perfect_info_gain = (local_frontier_info - local_pos_info) / local_frontier_info

        return perfect_info_gain

    def compute_utility(self, position: np.ndarray, frontier: np.ndarray) -> float:
        """
        Compute the utility value for the perfect information gain strategy.

        :param position: Current position of the robot as a 2D numpy array.
        :param frontier: 2D numpy array representing the frontier.
        :return: Utility value as a float.
        """
        # Calculate distance to the frontier
        distance_to_frontier = np.linalg.norm(frontier - position)

        # Calculate perfect information gain for the frontier
        perfect_info_gain = self.compute_perfect_information_gain(position, frontier)

        # Convert distance and perfect information gain to utility value
        utility_value = perfect_info_gain / (1.0 + distance_to_frontier)

        return utility_value

class ExplorationManager:
    """
    Class to manage the exploration process and select the next best location.
    """
    def __init__(self, map_size: Tuple[int, int], resolution: float):
        self.map_size = map_size
        self.resolution = resolution
        self.exploration_strategies = {
            'nearest_frontier': NearestFrontierStrategy(map_size, resolution),
            'information_gain': InformationGainStrategy(map_size, resolution),
            'perfect_information_gain': PerfectInformationGainStrategy(map_size, resolution)
        }
        self.current_position = None
        self.frontiers = None
        self.saliency_map = None

    def update_maps(self, occupied_map: np.ndarray, free_map: np.ndarray, saliency_map: np.ndarray) -> None:
        """
        Update the local maps used for exploration.

        :param occupied_map: 2D numpy array representing the occupied space.
        :param free_map: 2D numpy array representing the free space.
        :param saliency_map: 2D numpy array representing the saliency values.
        """
        for strategy in self.exploration_strategies.values():
            strategy.update_maps(occupied_map, free_map, saliency_map)

    def update_position(self, position: geom_msgs.Point) -> None:
        """
        Update the current position of the robot.

        :param position: Current position of the robot as a geometry_msgs.Point message.
        """
        self.current_position = np.array([position.x, position.y])

    def update_frontiers(self, frontiers: List[geom_msgs.Point]) -> None:
        """
        Update the list of detected frontiers.

        :param frontiers: List of geometry_msgs.Point messages representing the frontiers.
        """
        self.frontiers = np.array([[frontier.x, frontier.y] for frontier in frontiers])

    def update_saliency_map(self, saliency_map: np.ndarray) -> None:
        """
        Update the saliency map.

        :param saliency_map: 2D numpy array representing the saliency values.
        """
        self.saliency_map = saliency_map

    def select_next_location(self, strategy: str = 'nearest_frontier') -> np.ndarray:
        """
        Select the next best location to explore using the specified strategy.

        :param strategy: Exploration strategy to use (nearest_frontier, information_gain, perfect_information_gain).
        :return: Next location to explore as a 2D numpy array.
        """
        # Validate the chosen strategy
        if strategy not in self.exploration_strategies:
            raise ValueError(f"Invalid exploration strategy: {strategy}. Available strategies: {self.exploration_strategies.keys()}")

        # Get the selected strategy
        selected_strategy = self.exploration_strategies[strategy]

        # Validate current position
        if self.current_position is None:
            raise ValueError("Current position is not set.")

        # Validate frontiers
        if self.frontiers is None or len(self.frontiers) == 0:
            raise InvalidFrontierException("No valid frontiers found.")

        # Validate saliency map
        if self.saliency_map is None:
            raise ValueError("Saliency map is not set.")

        # Compute utility values for each frontier
        utility_values = np.zeros(self.frontiers.shape)
        for i, frontier in enumerate(self.frontiers):
            utility_values[i] = selected_strategy.compute_utility(self.current_position, frontier)

        # Apply saliency bias to the utility values
        biased_utility = selected_strategy.apply_saliency_bias(utility_values, self.saliency_map)

        # Select the best frontier based on the utility values
        next_location = selected_strategy.select_best_frontier(biased_utility)

        return next_location

# Example usage
if __name__ == "__main__":
    # Simulated maps and data
    occupied_map = np.zeros((200, 200))
    free_map = np.ones((200, 200))
    saliency_map = np.random.random((200, 200))

    # Simulated current position and frontiers
    current_position = geom_msgs.Point(x=5.0, y=5.0)
    frontiers = [geom_msgs.Point(x=10.0, y=10.0), geom_msgs.Point(x=15.0, y=15.0)]

    # Create an instance of ExplorationManager
    exploration_manager = ExplorationManager(map_size=(200, 200), resolution=MAP_RESOLUTION)

    # Update maps, position, frontiers, and saliency map
    exploration_manager.update_maps(occupied_map, free_map, saliency_map)
    exploration_manager.update_position(current_position)
    exploration_manager.update_frontiers(frontiers)
    exploration_manager.update_saliency_map(saliency_map)

    # Select the next location to explore using the nearest frontier strategy
    strategy = 'nearest_frontier'
    start_time = time.time()
    next_location = exploration_manager.select_next_location(strategy)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Selected next location using {strategy} strategy: {next_location}, Time taken: {elapsed_time:.2f}s")