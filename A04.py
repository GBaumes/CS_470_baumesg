# Assignment 04 - Extract LBP features from a set of images.
# Gregory Baumes
# 11/14/2023

from General_A04 import *
import cv2
import numpy as np

def getOneLBPLabel(subimage, label_type=LBP_LABEL_TYPES.UNIFORM):
    
    # The center pixel of the subimage
    center = subimage[1,1]
    # Hard code the outer neighbors of the center pixel
    outerValues = [subimage[0,0], subimage[0,1], subimage[0,2], subimage[1,2], subimage[2,2], subimage[2,1], subimage[2,0], subimage[1,0]]
    # Empty list to store the binary comparisons of the outerValues to the center pixel
    outerValuesToBinary = []
    # Loop over outerValues
    for pixel in outerValues:
        # If greater than the center value append a 1 onto list otherwise append a 0
        if pixel > center:
            outerValuesToBinary.append(1)
        else:
            outerValuesToBinary.append(0)
        
    # Get the rotation-invariant uniform LBP label
    transitionCount = 0
    currentBit = 0
    onesCount = 0
    # Loop over binary values
    for index, value in enumerate(outerValuesToBinary):
        # Check if index is the first element in the list if it is set currentBit equal to it
        if index == 0:
            currentBit = value
        # Check if index is the last element in the list and if so compare to first element.
        elif index == len(outerValuesToBinary) - 1:
            if value != outerValuesToBinary[0]:
                currentBit = value
                transitionCount += 1
        
        # Else if the value doesn't equal the current bit set current bit equal to the value and increment count by 1
        elif value != currentBit:
            currentBit = value
            transitionCount += 1
        
        # Count the number of 1s in the list
        if value == 1:
                onesCount += 1
        
    # Check to see if it is uniform    
    if transitionCount <= 2:
        label = onesCount
    else:
        label = len(outerValuesToBinary) + 1
        
    # Return label
    return label
    
def getLBPImage(image, label_type=LBP_LABEL_TYPES.UNIFORM):
    # Create a padded image with a 1 on all sides
    paddedImage = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    # Create and empty array to store outputs
    output = np.zeros_like(image)
    
    # Loop over each pixel and get subimage
    for i in range(1, image.shape[0] + 1):
        for j in range(1, image.shape[1] + 1):
            startRow = i - 1
            endRow = i + 2
            startCol = j - 1
            endCol = j + 2
            
            subimage = paddedImage[startRow:endRow, startCol:endCol]
            # Get the label for the subimage
            label = getOneLBPLabel(subimage)
            
            # Store the label in the output
            output[i-1, j-1] = label
    
    # Return the output image
    return output
    
def getOneRegionLBPFeatures(subimage, label_type=LBP_LABEL_TYPES.UNIFORM):
    # Initialize unnoramlized histogram as zero array shape 10 b/c uniform
    unnormalizedHist = np.zeros((10,), dtype=np.float32)
    # Loop over pizels and store in unnormalized hist
    for row in range(subimage.shape[0]):
        for column in range(subimage.shape[1]):
            currentPixel = subimage[row, column]
            unnormalizedHist[currentPixel] += 1
    # Initialize normalized histogram as zero array    
    normalizedHist = np.zeros((10,), dtype=np.float32)
    # Get the total sum of all pixels in the unnormalized histogram
    total = np.sum(unnormalizedHist)
    # Loop over unnormalized histogram and store normalized
    for i in range(unnormalizedHist.shape[0]):
        normalizedHist[i] = unnormalizedHist[i]/total
    
    # Return the normalized histogram
    return normalizedHist

    
def getLBPFeatures(featureImage, regionSideCnt, label_type=LBP_LABEL_TYPES.UNIFORM):
    # Get the dimension of the featuredImage
    imageHeight = featureImage.shape[0]
    imageWidth = featureImage.shape[1]
    
    # Get the subregions width and height
    subHeight = imageHeight // regionSideCnt # Floor division
    subWidth = imageWidth // regionSideCnt
    
    # Create empty list for histograms
    allHists = []
    
    # Loop through subregions
    for i in range(regionSideCnt):
        for j in range(regionSideCnt):
            # Starting point for each subregion
            startRow = i * subHeight
            startCol = j * subWidth
            
            # Get subimage
            subImage = featureImage[startRow:startRow + subHeight, startCol: startCol + subWidth]
            
            # getOneRegionLBPFeatures
            histogram = getOneRegionLBPFeatures(subImage)
            
            # Append to list
            allHists.append(histogram)
            
    # Convert list to array and reshape
    allHists = np.array(allHists)
    allHists = np.reshape(allHists, (allHists.shape[0]*allHists.shape[1],))
    
    return allHists