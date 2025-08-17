import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from typing import Tuple, Optional

# Define constants
MAP_RESOLUTION = 0.1  # meters per pixel
MAP_SIZE = (512, 512)  # pixels
CNN_INPUT_SIZE = (224, 224)  # pixels
CNN_INPUT_CHANNELS = 3  # RGB

# Define logger
logger = logging.getLogger(__name__)

class MapProcessor:
    """
    Handles map preprocessing, conversion between ROS occupancy grids and CNN input format, and map visualization utilities.
    """

    def __init__(self, map_resolution: float = MAP_RESOLUTION, map_size: Tuple[int, int] = MAP_SIZE):
        """
        Initializes the MapProcessor.

        Args:
        - map_resolution (float): The resolution of the map in meters per pixel. Defaults to MAP_RESOLUTION.
        - map_size (Tuple[int, int]): The size of the map in pixels. Defaults to MAP_SIZE.
        """
        self.map_resolution = map_resolution
        self.map_size = map_size
        self.cv_bridge = CvBridge()

    def grid_to_image(self, grid: OccupancyGrid) -> np.ndarray:
        """
        Converts a ROS occupancy grid to a numpy image.

        Args:
        - grid (OccupancyGrid): The ROS occupancy grid.

        Returns:
        - image (np.ndarray): The numpy image representation of the grid.
        """
        try:
            # Convert grid to numpy array
            grid_data = np.array(grid.data, dtype=np.uint8)
            grid_data = grid_data.reshape(grid.info.height, grid.info.width)

            # Create a 3-channel image (RGB)
            image = np.zeros((grid.info.height, grid.info.width, 3), dtype=np.uint8)

            # Set pixel values based on occupancy
            image[grid_data == 0] = (255, 255, 255)  # Free space
            image[grid_data == 100] = (0, 0, 0)  # Occupied space
            image[grid_data == -1] = (128, 128, 128)  # Unknown space

            return image
        except Exception as e:
            logger.error(f"Error converting grid to image: {e}")
            return None

    def image_to_grid(self, image: np.ndarray) -> OccupancyGrid:
        """
        Converts a numpy image to a ROS occupancy grid.

        Args:
        - image (np.ndarray): The numpy image representation of the grid.

        Returns:
        - grid (OccupancyGrid): The ROS occupancy grid.
        """
        try:
            # Create a ROS occupancy grid
            grid = OccupancyGrid()
            grid.info.resolution = self.map_resolution
            grid.info.width = image.shape[1]
            grid.info.height = image.shape[0]
            grid.info.origin.position.x = 0
            grid.info.origin.position.y = 0
            grid.info.origin.position.z = 0
            grid.info.origin.orientation.x = 0
            grid.info.origin.orientation.y = 0
            grid.info.origin.z = 0
            grid.info.origin.w = 1

            # Convert image to grid data
            grid_data = np.zeros((grid.info.height, grid.info.width), dtype=np.uint8)
            grid_data[(image[:, :, 0] == 255) & (image[:, :, 1] == 255) & (image[:, :, 2] == 255)] = 0  # Free space
            grid_data[(image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)] = 100  # Occupied space
            grid_data[(image[:, :, 0] == 128) & (image[:, :, 1] == 128) & (image[:, :, 2] == 128)] = -1  # Unknown space

            # Flatten grid data
            grid.data = grid_data.flatten().tolist()

            return grid
        except Exception as e:
            logger.error(f"Error converting image to grid: {e}")
            return None

    def preprocess_for_cnn(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses a numpy image for input to a CNN.

        Args:
        - image (np.ndarray): The numpy image to preprocess.

        Returns:
        - preprocessed_image (np.ndarray): The preprocessed numpy image.
        """
        try:
            # Resize image to CNN input size
            resized_image = cv2.resize(image, (CNN_INPUT_SIZE[0], CNN_INPUT_SIZE[1]))

            # Normalize pixel values to [0, 1]
            normalized_image = resized_image / 255.0

            return normalized_image
        except Exception as e:
            logger.error(f"Error preprocessing image for CNN: {e}")
            return None

    def visualize_map(self, grid: OccupancyGrid) -> None:
        """
        Visualizes a ROS occupancy grid using matplotlib.

        Args:
        - grid (OccupancyGrid): The ROS occupancy grid to visualize.
        """
        try:
            # Convert grid to image
            image = self.grid_to_image(grid)

            # Display image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        except Exception as e:
            logger.error(f"Error visualizing map: {e}")

    def save_map_snapshot(self, grid: OccupancyGrid, filename: str) -> None:
        """
        Saves a snapshot of a ROS occupancy grid to a file.

        Args:
        - grid (OccupancyGrid): The ROS occupancy grid to save.
        - filename (str): The filename to save the snapshot to.
        """
        try:
            # Convert grid to image
            image = self.grid_to_image(grid)

            # Save image to file
            cv2.imwrite(filename, image)
        except Exception as e:
            logger.error(f"Error saving map snapshot: {e}")

    def compute_map_coverage(self, grid: OccupancyGrid) -> float:
        """
        Computes the coverage of a ROS occupancy grid.

        Args:
        - grid (OccupancyGrid): The ROS occupancy grid to compute coverage for.

        Returns:
        - coverage (float): The coverage of the grid as a percentage.
        """
        try:
            # Convert grid to image
            image = self.grid_to_image(grid)

            # Count number of free space pixels
            free_space_pixels = np.sum((image[:, :, 0] == 255) & (image[:, :, 1] == 255) & (image[:, :, 2] == 255))

            # Compute coverage
            coverage = (free_space_pixels / (image.shape[0] * image.shape[1])) * 100

            return coverage
        except Exception as e:
            logger.error(f"Error computing map coverage: {e}")
            return 0.0

class MapProcessorException(Exception):
    """
    Custom exception class for MapProcessor.
    """

    def __init__(self, message: str):
        """
        Initializes the MapProcessorException.

        Args:
        - message (str): The error message.
        """
        self.message = message
        super().__init__(self.message)

def main():
    # Create a MapProcessor instance
    map_processor = MapProcessor()

    # Create a sample ROS occupancy grid
    grid = OccupancyGrid()
    grid.info.resolution = 0.1
    grid.info.width = 512
    grid.info.height = 512
    grid.info.origin.position.x = 0
    grid.info.origin.position.y = 0
    grid.info.origin.position.z = 0
    grid.info.origin.orientation.x = 0
    grid.info.origin.orientation.y = 0
    grid.info.origin.z = 0
    grid.info.origin.w = 1
    grid.data = [0] * (512 * 512)

    # Visualize the grid
    map_processor.visualize_map(grid)

    # Save a snapshot of the grid
    map_processor.save_map_snapshot(grid, "map_snapshot.png")

    # Compute the coverage of the grid
    coverage = map_processor.compute_map_coverage(grid)
    print(f"Map coverage: {coverage}%")

if __name__ == "__main__":
    main()