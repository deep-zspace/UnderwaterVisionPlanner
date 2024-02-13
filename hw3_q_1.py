#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


class ImageProcessor:

    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    @staticmethod
    def robust_normalizer(image, clahe_clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.2):
        if len(image.shape) == 2 or image.shape[2] == 1:  # Check if grayscale
            # Convert grayscale to BGR (3-channel)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])

        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=tile_grid_size)
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])

        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # Applying Gamma Correction
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma *
                         255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(result, table)

        return result

    def detect_and_match_features(self, image1, image2):
        keypoints1, descriptors1 = self.sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(image2, None)

        matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [
            m for m, n in matches if m.distance < 0.75 * n.distance]

        return keypoints1, keypoints2, good_matches

    @staticmethod
    def compute_homography_and_filter_matches(kp1, kp2, matches, threshold=5.0):
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Check if there are enough matches
        if len(src_pts) < 4 or len(dst_pts) < 4:
            print("Warning: Not enough matches to compute the homography.")
            return None, []

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
        matches_inliers = [matches[i]
                           for i, val in enumerate(mask) if val == 1]
        return H, matches_inliers

    @staticmethod
    def objective_function(H_flat, src_points, dst_points):
        """Objective function to compute reprojection error."""
        H = H_flat.reshape(3, 3)
        projected_points = ImageProcessor.project_points(H, src_points)
        return (projected_points - dst_points).ravel()

    @staticmethod
    def project_points(H, points):
        """Projects points using a homography matrix."""
        homogeneous_points = np.column_stack(
            [points, np.ones(points.shape[0])])
        transformed_homogeneous = homogeneous_points.dot(H.T)
        return (transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:3])

    @staticmethod
    def refine_homography_with_LM(src_points, dst_points, initial_H):
        result = least_squares(ImageProcessor.objective_function, initial_H.ravel(
        ), args=(src_points, dst_points), method='lm')
        return result.x.reshape(3, 3)

    @staticmethod
    def is_valid_homography(H, matches_inliers, determinant_threshold=1e-6, inliers_threshold=10):
        det_H = np.linalg.det(H)
        if abs(det_H) < determinant_threshold:
            return False
        if len(matches_inliers) < inliers_threshold:
            return False
        return True

    def store_valid_and_refined_homographies(self, image_list, image_pairs):
        homographies = {}
        for (src_idx, dst_idx) in image_pairs:
            img1 = image_list[src_idx - 1]
            img2 = image_list[dst_idx - 1]

            kp1, kp2, good_matches = self.detect_and_match_features(img1, img2)
            H, matches_inliers = self.compute_homography_and_filter_matches(
                kp1, kp2, good_matches)

            if self.is_valid_homography(H, matches_inliers):
                H_refined = self.refine_homography_with_LM(
                    np.float32(
                        [kp1[m.queryIdx].pt for m in matches_inliers]).reshape(-1, 2),
                    np.float32(
                        [kp2[m.trainIdx].pt for m in matches_inliers]).reshape(-1, 2),
                    H
                )
                key = f'H_{src_idx}{dst_idx}'
                homographies[key] = H_refined
        return homographies

    @staticmethod
    def decompose_homography(H):
        dx, dy = H[0, 2], H[1, 2]
        theta = np.arctan2(H[1, 0] , H[0, 0] )
        dx = dx*(-1)
        return dx, dy, theta

    def process_image_pairs(self, image_list, pairs=None):
        if pairs is None:
            pairs = [(i, i+1) for i in range(1, len(image_list))]
        homographies = self.store_valid_and_refined_homographies(image_list, pairs)

        motions = {}
        for key, H in homographies.items():
            dx, dy, theta = self.decompose_homography(H)
            motions[f'pose_{key.split("_")[1]}'] = {'x': dx, 'y': dy, 'theta': theta}

            # For visualization: plot the keypoint matches for this pair
            src_idx, dst_idx = int(key.split("_")[1][0]), int(key.split("_")[1][1])
            img1 = image_list[src_idx - 1]
            img2 = image_list[dst_idx - 1]
            kp1, kp2, good_matches = self.detect_and_match_features(img1, img2)
            _, matches_inliers = self.compute_homography_and_filter_matches(kp1, kp2, good_matches)
            self.plot_keypoint_matches(img1, kp1, img2, kp2, matches_inliers, f"Image {src_idx}", f"Image {dst_idx}")

        return homographies, motions


    def compute_absolute_poses(self, motions, image_list):
        absolute_poses = {}
        absolute_poses = {'pose_1': {'x': 0, 'y': 0, 'theta': 0}}
        for i in range(1, len(image_list)):
            prev_pose = absolute_poses[f'pose_{i}']
            relative_motion = motions.get(f'pose_{i}{i + 1}', None)
            if relative_motion is None:
                # Added print statement
                print(f"Relative motion for pose_{i} to pose_{i+1} is missing")
                # If relative motion is missing, continue with the next image
                continue
            dx = relative_motion['x']
            dy = relative_motion['y']
            dtheta = relative_motion['theta']
            new_x = prev_pose['x'] + dx * \
                np.cos(prev_pose['theta']) - dy * np.sin(prev_pose['theta'])
            new_y = prev_pose['y'] + dx * \
                np.sin(prev_pose['theta']) + dy * np.cos(prev_pose['theta'])
            new_theta = prev_pose['theta'] + dtheta
            absolute_poses[f'pose_{i + 1}'] = {'x': new_x,
                                               'y': new_y, 'theta': new_theta}
        return absolute_poses

    def plot_absolute_poses(poses):
        # Extract x, y, and theta values
        x = [pose['x'] for pose in poses.values()]
        y = [pose['y'] for pose in poses.values()]
        theta = [pose['theta'] for pose in poses.values()]

        # Plot the trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, '-o', label='Trajectory')

        # Optionally plot arrows indicating the orientation at each pose
        arrow_length = 0.5
        for xi, yi, thetai in zip(x, y, theta):
            dx = arrow_length * np.cos(thetai)
            dy = arrow_length * np.sin(thetai)
            plt.arrow(xi, yi, dx, dy, head_width=0.1,
                      head_length=0.15, fc='red', ec='red')

        # Labels and legend
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Absolute Poses')
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_images(images, titles=None):
        num_images = len(images)
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
        if num_images == 1:
            axs = [axs]
        for i, ax in enumerate(axs):
            ax.imshow(images_rgb[i])
            ax.axis('off')
            if titles:
                ax.set_title(titles[i])
        plt.tight_layout()
        plt.show()
        
    def plot_keypoint_matches(self, img1, kp1, img2, kp2, matches_inliers, img1_name, img2_name):
        """
        Plot the keypoint matches of two images after RANSAC filtering.
        """
        matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches_inliers, None)
        plt.figure(figsize=(15,7))
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Key Point Matches between {img1_name} and {img2_name}')
        plt.show()

    @staticmethod
    def alpha_blending(img1, img2):
        channels = img1.shape[2] if len(img1.shape) == 3 else 1
        if channels == 3:
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([5, 5, 5])
        else:
            lower_bound = np.array([0])
            upper_bound = np.array([5])

        mask1 = cv2.inRange(img1, lower_bound, upper_bound)
        mask2 = cv2.inRange(img2, lower_bound, upper_bound)
        mask1 = cv2.bitwise_not(mask1)
        mask2 = cv2.bitwise_not(mask2)
        mask = cv2.bitwise_and(mask1, mask2)
        mask_inv = cv2.bitwise_not(mask)
        img1_fg = cv2.bitwise_and(img1, img1, mask=mask)
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
        img_fg = cv2.addWeighted(img1_fg, 0.7, img2_fg, 0.3, 0)
        img1_bg = cv2.bitwise_and(img1, img1, mask=cv2.bitwise_and(mask1, mask_inv))
        img2_bg = cv2.bitwise_and(img2, img2, mask=cv2.bitwise_and(mask2, mask_inv))
        img = cv2.add(img2_bg, img1_bg)
        img = cv2.add(img, img_fg)
        return img

    @staticmethod
    def warpTwoImages(img1, img2, H):
        """
        warp img2 to img1 with homograph H
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        pts1 = np.float32([[[0, 0], [0, h1], [w1, h1], [w1, 0]]]).reshape(-1, 1, 2)
        pts2 = np.float32([[[0, 0], [0, h2], [w2, h2], [w2, 0]]]).reshape(-1, 1, 2)
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts2, pts1_), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        shift = pts1_[0][0]
        Ht = np.asarray([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
        result = cv2.warpPerspective(img1, Ht.dot(H), (xmax - xmin, ymax - ymin))
        H_t = np.dot(np.eye(3), np.asarray([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]]))
        tmp = cv2.warpPerspective(img2, H_t, (xmax - xmin, ymax - ymin))
        output = ImageProcessor.alpha_blending(result, tmp)
        return output, t


if __name__ == "__main__":

    # Load the images (assuming they are named image1.jpg, image2.jpg, etc. and are located in the current directory)
    images = [cv2.imread(f"/home/deep/data/NEU/AFR/eece7150/hw3/6Images/{i}.tif") for i in range(1, 7)]
    if any(img is None for img in images):
        print("Error loading one or more images.")
        exit()

    # Initialize the processor
    processor = ImageProcessor()

    # Robust normalization of images (optional, only if required)
    images = [processor.robust_normalizer(img) for img in images]
    ImageProcessor.plot_images(
        images, titles=[f"Image {i+1}" for i in range(len(images))])

    # Process sequential image pairs
    homographies, motions = processor.process_image_pairs(images)
    print("Homographies (Sequential):", homographies)
    print("Motions (Sequential):", motions)

    # Process non-sequential image pairs
    pairs = [(1, 6), (2, 5), (2, 4), (3, 5), (1, 5), (2, 6)]
    non_sequential_homographies, non_sequential_motions = processor.process_image_pairs(
        images, pairs)
    print("Homographies (Non-Sequential):", non_sequential_homographies)
    print("Motions (Non-Sequential):", non_sequential_motions)

    # Compute absolute poses
    absolute_poses = processor.compute_absolute_poses(motions, images)
    print(motions)
    print("Absolute Poses:", absolute_poses)
    ImageProcessor.plot_absolute_poses(absolute_poses)
    
    # Iterate over all valid homography pairs and create mosaics
    for key, H in homographies.items(): 
        src_idx, dst_idx = int(key.split("_")[1][0]), int(key.split("_")[1][1])
        img1 = images[src_idx - 1]
        img2 = images[dst_idx - 1]

        if processor.is_valid_homography(H, [], inliers_threshold=0):  # setting inliers_threshold to 0 since we don't have inliers here
            mosaic, _ = processor.warpTwoImages(img1, img2, H)
 
            # Crop black borders
            gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            crop = mosaic[y:y+h, x:x+w]

            # Display the mosaic using matplotlib
            plt.figure(figsize=(10,5))
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plt.title(f"Mosaic between Image {src_idx} and Image {dst_idx}")
            plt.axis('off')
            plt.show()

