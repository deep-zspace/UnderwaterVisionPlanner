import math
import os
import gtsam
import numpy as np
import cv2
from hw3_q_1 import ImageProcessor
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot

class GTSAMOptimizer:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.result = None
        self.ERF_DIV = 45
        self.LIN_DIV = 25

    # Primary Methods
    def add_prior_factor(self, pose_id=1, prior_mean=gtsam.Pose2(0.0, 0.0, 0.0), noise_sigmas=np.array([0.1, 0.1, 0.1])):
        """
        Add a prior factor to the graph.
        """
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
        self.graph.add(gtsam.PriorFactorPose2(pose_id, prior_mean, prior_noise))
        if not self.initial_estimates.exists(pose_id):  # Check if key exists
            self.initial_estimates.insert(pose_id, prior_mean)

    def add_in_between_factors(self, motions):
        return self._add_relative_factors(motions)

    def add_loop_closure_factors(self, motions):
        return self._add_relative_factors(motions, False)

    def _add_relative_factors(self, motions, is_sequential=True):
        for key, motion in motions.items():
            import re
            pose_ids = re.findall(r'\d+', key)
            pose_id_from, pose_id_to = map(int, pose_ids)

            relative_pose = gtsam.Pose2(motion['x'], motion['y'], motion['theta'])
            matches = motion.get('matches', 10)
            covariance = self.compute_covariance(motion['x'], motion['y'], motion['theta'], matches)
            noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
            self.graph.add(gtsam.BetweenFactorPose2(pose_id_from, pose_id_to, relative_pose, noise))
        print("\nFactor Graph:\n{}".format(self.graph))
        
    def create_initial_estimate(self, absolute_poses):
        for key, pose in absolute_poses.items():
            pose_id = int(key.split('pose_')[1])
            initial_pose = gtsam.Pose2(pose['x'], pose['y'], pose['theta'])
            if not self.initial_estimates.exists(pose_id):
                self.initial_estimates.insert(pose_id, initial_pose)
        print("\nInitial Estimate:\n{}".format(self.initial_estimates))

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates, params)
        self.result = optimizer.optimize()
        print("\nFinal Result:\n{}".format(self.result))
        error = self.graph.error(self.result)
        print(f"Optimization complete. Final error = {error:.2f}")

    def extract_optimized_poses(self):
        poses = {}
        for key in self.result.keys():
            pose = self.result.atPose2(key)
            poses[f"pose_{key}"] = {'x': pose.x(), 'y': pose.y(), 'theta': pose.theta()}
        return poses

    # Visualization Methods
    def print_marginal_covariances(self):
        if self.result is None:
            return
        marginals = gtsam.Marginals(self.graph, self.result)
        for key in self.result.keys():
            covariance = marginals.marginalCovariance(key)
            print(f"X{key} covariance:\n{covariance}\n")

    def compute_covariance(self, dx, dy, dth, matches):
        """Compute covariance matrix based on relative motion and number of matches."""
        matches = min(matches, 30)
        covar_multiplier = 1 / (1 + math.erf((matches - 3) / self.ERF_DIV))
        min_covar_value = 0.01  # minimum covariance value
        exx = max(abs(dx / self.LIN_DIV) ** 2, min_covar_value) * covar_multiplier
        eyy = max(abs(dy / self.LIN_DIV) ** 2, min_covar_value) * covar_multiplier
        ett = max(abs(dth / (self.LIN_DIV * 10.0)) ** 2, min_covar_value) * covar_multiplier
        exy = 0.01 * covar_multiplier
        ext = 0.01 * covar_multiplier
        eyt = 0.01 * covar_multiplier

        covariance = np.array([[exx, exy, ext], [eyt, eyy, eyt], [ext, exy, ett]])
        return covariance

    def plot_trajectories(self, before_poses, after_poses, title='Trajectories Before and After Optimization'):
        x_before = [pose['x'] for pose in before_poses.values()]
        y_before = [pose['y'] for pose in before_poses.values()]
        x_after = [pose['x'] for pose in after_poses.values()]
        y_after = [pose['y'] for pose in after_poses.values()]

        plt.figure(figsize=(10, 6))
        plt.plot(x_before, y_before, '-o', label='Before Optimization')
        plt.plot(x_after, y_after, '-x', label='After Optimization')

        for pose in before_poses.values():
            gtsam_pose = gtsam.Pose2(pose['x'], pose['y'], pose['theta'])
            gtsam_plot.plot_pose2(1,            gtsam_pose, 20)  # scale of 0.5 for the axes

        for pose in after_poses.values():
            gtsam_pose = gtsam.Pose2(pose['x'], pose['y'], pose['theta'])
            gtsam_plot.plot_pose2(1, gtsam_pose, 20)  # scale of 0.5 for the axes

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def add_initial_estimate(self, key, pose):
        """
        Add an initial estimate for a pose to the initial estimates.
        """
        int_key = int(key.split('_')[1])  # Extracting the integer from the string key
        if not self.initial_estimates.exists(int_key):  # Check if key exists
            self.initial_estimates.insert(int_key, pose)

    def plot_covariances(self, values, title="Pose Covariances"):
        marginals = gtsam.Marginals(self.graph, values)
        plt.figure(figsize=(10, 6))
        plt.title(title)

        for i in range(1, 7):  # Assuming you have 6 poses, adjust if needed
            if values.exists(i):  # Check if the pose exists in the values
                pose = values.atPose2(i)
                gtsam_plot.plot_pose2(1, pose, 50, marginals.marginalCovariance(i))
                plt.text(pose.x(), pose.y() + 0.2, f'Pose {i}', ha='center', va='center', color='blue')

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def plot_factor_graph(self):
        """Visualize the GTSAM factor graph using Graphviz."""
        
        # Create a GraphvizFormatting object
        formatting = gtsam.GraphvizFormatting()
        
        # Customize the formatting as needed
        formatting.figureWidthInches = 10
        formatting.figureHeightInches = 10
        formatting.scale = 1.0
        formatting.plotFactorPoints = True
        formatting.connectKeysToFactor = True
        
        # Save the graph to a dot file
        dot_file = "/tmp/factor_graph.dot"
        
        # Use the self.result as the Values object for generating Graphviz representation
        self.graph.saveGraph(dot_file, self.result, keyFormatter=gtsam.DefaultKeyFormatter, writer=formatting)
        
        # Convert dot to PNG
        os.system(f"dot -Tpng {dot_file} -o /tmp/factor_graph.png")
        
        # Display the PNG
        img = plt.imread("/tmp/factor_graph.png")
        plt.imshow(img)
        plt.axis('on')  # Hide axes
        plt.show()

def main():
    processor = ImageProcessor()
    optimizer = GTSAMOptimizer()

    images = [cv2.imread(f"./6Images/{i}.tif") for i in range(1, 7)]
    if any(img is None for img in images):
        print("Error loading one or more images.")
        return

    images = [processor.robust_normalizer(img) for img in images]

    homographies, motions = processor.process_image_pairs(images)
    pairs = [(1, 6), (2, 5), (2, 4), (3, 5), (1, 5), (2, 6)]
    non_sequential_homographies, non_sequential_motions = processor.process_image_pairs(images, pairs)

    optimizer.add_prior_factor()
    optimizer.add_in_between_factors(motions)
    optimizer.add_loop_closure_factors(non_sequential_motions)
    absolute_poses_before = processor.compute_absolute_poses(motions, images)
    optimizer.create_initial_estimate(absolute_poses_before)
    
    
    optimizer.optimize()
    absolute_poses_after = optimizer.extract_optimized_poses()
    optimizer.print_marginal_covariances()
    optimizer.plot_factor_graph()

    optimizer.plot_trajectories(absolute_poses_before, absolute_poses_after)
    optimizer.plot_covariances(optimizer.initial_estimates, "Covariances Before Optimization")
    optimizer.plot_covariances(optimizer.result, "Covariances After Optimization")
    
if __name__ == "__main__":
    main()

