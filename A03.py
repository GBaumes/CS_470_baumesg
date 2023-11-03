# A03 - Assignment 3 for CS 470. Detecting the bounding boxes for cells in the BCCD dataset.

import skimage as ski
import numpy as np
import cv2

def find_WBC(image):
    '''
        Takes in a color image. 
        Finds the white blood cells.
        Computes the bounding box for each white blood cell found.
        Returns a list of detected bounding boxes.
    '''
    # Get superpixel groups.
    segments = ski.segmentation.slic(image, n_segments=50, sigma=5, start_label=0)
    
    # Calculate the number of superpixel groups 
    cnt = len(np.unique(segments))
    
    # Compute mean color per group
    group_means = np.zeros((cnt, 3), dtype="float32")
    
    # Loop through each superpixel
    for specific_group in range(cnt):
        # Create mask images
        mask_image = np.where(segments==specific_group, 255, 0).astype("uint8")
        # Add channel dimension back into the mask
        mask_image = np.expand_dims(mask_image, axis=2)
        # Compute the mean value per group
        group_means[specific_group] = cv2.mean(image, mask=mask_image)[0:3]
    
    # Use K-means on GROUP mean colors to group them into 4 color groups.
    # Number of desired clusters
    k = 4
    ret, bestLabels, centers = cv2.kmeans(data=group_means, 
                                         K=k,
                                         bestLabels=None,
                                         criteria=(
                                             cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                             10, 1.0),
                                         attempts=10,
                                         flags=cv2.KMEANS_RANDOM_CENTERS)
    
    # White blood cells color value (BGR in openCV)
    WBC = np.array([255,0,0])
            
    # numpy constant for positive infinity
    min_distance = np.inf
    # initialize closest_group to the first group
    closest_group = 0
    
    # Loop through the number of clusters
    for i in range(k):
        # Find distance from the target color
        distance = np.sqrt(np.sum((centers[i] - WBC)**2))
        # If the distance is less than positive infinity set the min_distance equal to that distance, and set the closest group equal to i (the current cluster)
        if distance < min_distance:
            min_distance = distance
            closest_group = i
        
    # Loop through number of clusters. Set the closest_group to white and the rest of the clusters to black
    for i in range(k):
        if i == closest_group:
            centers[i] = [255,255,255]
        else:
            centers[i] = [0,0,0]
    
    # Convert the centers to unsigned 8-bit
    centers = centers.astype(np.uint8)
    # Get the new superpixel group colors
    colors_per_clump = centers[bestLabels.flatten()]
    
    # Create a new image
    cell_mask = colors_per_clump[segments]
    
    # Convert to grayscale
    cell_mask_gray = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
    
    # Get disjoint blobs from cell_mask
    retval, labels = cv2.connectedComponents(cell_mask_gray, connectivity=4)
    
    # Create empty list to store bounding boxes.
    bounding_boxes = []
    
    # For each blob group except 0 get the coords of the pixel and ass those coords to the bounding_boxes list.
    for i in range(1, retval):
        coords = np.where(labels == i)
        if len(coords[0]) > 0:
            ymin, xmin = np.min(coords, axis=1)
            ymax, xmax = np.max(coords, axis=1)
            bounding_boxes.append((ymin, xmin, ymax, xmax))
        
    # Return bounding_boxes list
    return bounding_boxes