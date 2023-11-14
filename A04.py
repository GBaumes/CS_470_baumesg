# Assignment 04 - Extract LBP features from a set of images.
# Gregory Baumes
# 11/14/2023

from General_A04 import *

def getOneLBPLabel(subimage, label_type):
    
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
    
def getLBPImage(image, label_type):
    ''''''
    
def getOneRegionLBPFeatures(subimage, label_type):
    ''''''
    
def getLBPFeatures(featureImage, regionSideCnt, label_type):
    ''''''