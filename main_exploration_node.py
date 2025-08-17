import rospy
import nav_msgs
import geometry_msgs
import tf2_ros
import actionlib
import cv_bridge
import message_filters
import yaml
import numpy as np
from typing import Tuple, List
from model_architecture import CNN_Map_Completion_Classifier
from saliency_computation import Grad_CAM_Saliency
from frontier_detection import Frontier_Detection
from exploration_strategies import Nearest_Frontier, Information_Gain, Perfect_Information_Gain
from config import Config

class ExplorationNode:
    """
    ROS node orchestrating the complete exploration pipeline.
    """
    def __init__(self):
        """
        Initialize the exploration node.
        """
        self.config = Config()
        self.map_sub = message_filters.Subscriber('/map', nav_msgs.msg.OccupancyGrid)
        self.map_pub = rospy.Publisher('/map', nav_msgs.msg.OccupancyGrid, queue_size=10)
        self.goal_pub = rospy.Publisher('/goal', geometry_msgs.msg.PoseStamped, queue_size=10)
        self.saliency_map_pub = rospy.Publisher('/saliency_map', nav_msgs.msg.OccupancyGrid, queue_size=10)
        self.frontier_detection = Frontier_Detection()
        self.saliency_computation = Grad_CAM_Saliency()
        self.cnn_map_completion_classifier = CNN_Map_Completion_Classifier()
        self.exploration_strategies = {
            'nearest_frontier': Nearest_Frontier(),
            'information_gain': Information_Gain(),
            'perfect_information_gain': Perfect_Information_Gain()
        }
        self.current_strategy = self.config.get('exploration_strategy')
        self.map = None
        self.saliency_map = None
        self.frontiers = None
        self.goal = None
        self.log_file = open('exploration_log.txt', 'w')

    def map_callback(self, map_msg: nav_msgs.msg.OccupancyGrid):
        """
        Update the map and trigger the exploration loop.
        """
        self.map = map_msg
        self.run_exploration_loop()

    def run_exploration_loop(self):
        """
        Run the exploration loop.
        """
        try:
            # Update saliency map
            self.update_saliency_map()
            # Detect frontiers
            self.frontiers = self.frontier_detection.detect_frontiers(self.map, self.saliency_map)
            # Select next frontier
            self.goal = self.select_next_frontier()
            # Publish goal
            self.publish_goal()
            # Log metrics
            self.log_metrics()
        except Exception as e:
            rospy.logerr(f'Error in exploration loop: {e}')

    def select_next_frontier(self) -> geometry_msgs.msg.PoseStamped:
        """
        Select the next frontier based on the current strategy.
        """
        try:
            strategy = self.exploration_strategies[self.current_strategy]
            return strategy.select_next_frontier(self.frontiers, self.map, self.saliency_map)
        except Exception as e:
            rospy.logerr(f'Error in selecting next frontier: {e}')
            return None

    def publish_goal(self):
        """
        Publish the goal pose.
        """
        try:
            self.goal_pub.publish(self.goal)
        except Exception as e:
            rospy.logerr(f'Error in publishing goal: {e}')

    def update_saliency_map(self):
        """
        Update the saliency map using the Grad-CAM saliency computation.
        """
        try:
            self.saliency_map = self.saliency_computation.compute_saliency_map(self.map)
            self.saliency_map_pub.publish(self.saliency_map)
        except Exception as e:
            rospy.logerr(f'Error in updating saliency map: {e}')

    def log_metrics(self):
        """
        Log exploration metrics.
        """
        try:
            metrics = {
                'frontiers': len(self.frontiers),
                'saliency_map': self.saliency_map,
                'goal': self.goal
            }
            self.log_file.write(f'{metrics}\n')
            self.log_file.flush()
        except Exception as e:
            rospy.logerr(f'Error in logging metrics: {e}')

    def shutdown_hook(self):
        """
        Shutdown hook to close the log file.
        """
        self.log_file.close()

def main():
    rospy.init_node('exploration_node')
    node = ExplorationNode()
    rospy.Subscriber('/map', nav_msgs.msg.OccupancyGrid, node.map_callback)
    rospy.on_shutdown(node.shutdown_hook)
    rospy.spin()

if __name__ == '__main__':
    main()