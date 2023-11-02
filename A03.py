# A03 - Assignment 3 for CS 470. Detecting the bounding boxes for cells in the BCCD dataset.

from skimage.segmentation import slic
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
    segments = slic(image, n_segments=100, sigma=5, start_label=0)
    
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
    
    # # Calculate the first center
    # first_center = centers[0]
    # min_distance = np.sqrt(np.sum((first_center - WBC)**2))
    # closest_center = 0
    
    # # Loop through rest of centers to find closest one
    # for i, center in enumerate(centers[1:]):
    #     distance = np.sqrt(np.sum((center - WBC) ** 2))
    #     if distance < min_distance:
    #         min_distance = distance
    #         closest_center = i
    
    # new_centers = centers.copy()
    
    # for i, center in enumerate(centers):
    #     if np.array_equal(center, centers[closest_center]):
    #         new_centers[i] = np.array([255,255,255])
    #     else:
    #         new_centers[i] = np.array([0,0,0])
            
    # centersUint8 = new_centers.astype(np.uint8)
    # colors_per_clump = centersUint8[bestLabels.flatten()]
    
    # cell_mask = colors_per_clump[segments]
    # cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
    
    # retval, labels = cv2.connectedComponents(cell_mask)
    
    # bounding_boxes = []
    # for i in range(1, retval):
    #     coords = np.where(labels == i)
    #     if coords:
    #         ymin, xmin, ymax, xmax = coords[0][0], coords[1][0], coords[0][-1], coords[1][-1]
    #         bounding_boxes.append((ymin, xmin, ymax, xmax))
            
    
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    min_distance = float('inf')
    closest_group = 0
    
    for i in range(k):
        distance = np.linalg.norm(centers[i] - WBC)
        if distance < min_distance:
            min_distance = distance
            closest_group = 1
        
    for i in range(k):
        if i == closest_group:
            centers[i] = [255,255,255]
        else:
            centers[i] = [0,0,0]
    
    centers = centers.astype(np.uint8)
    colors_per_clump = centers[bestLabels.flatten()]
    
    cell_mask = colors_per_clump[segments]
    
    cell_mask_gray = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
    
    retval, labels = cv2.connectedComponents(cell_mask_gray, connectivity=4)
    
    bounding_boxes = []
    
    for i in range(1, retval):
        coords = np.where(labels == i)
        if len(coords[0]) > 0:
            ymin, xmin = np.min(coords, axis=1)
            ymax, xmax = np.max(coords, axis=1)
            bounding_boxes.append((ymin, xmin, ymax, xmax))
            
    return bounding_boxes