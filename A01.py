# Assignment 1 Code
# CS 470
# Written by Gregory Baumes
# 9/27/2023

import gradio as gr
import cv2
import numpy as np

def create_unnormalized_hist(image):   
    # Create a zeroed out numpy array of shape (256,) and type float32.
    unnormalizedHist = np.zeros((256,), dtype=np.float32)

    # Loop over each pixel in the image, store the value of the current pixel in a variable,
    # Then add one the the position of the value of the current pixel in the unnormalized histogram.
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            currentPixel = image[row, column]
            unnormalizedHist[currentPixel] = unnormalizedHist[currentPixel] + 1
    
    # Return the unnormalized histogram.
    return unnormalizedHist
    
def normalize_hist(hist):
    # Get the sum of all the elements in the histogram
    total = np.sum(hist)
    # Create a zeroed out array to store the values of the normalized histogram in.
    normalizedHist = np.zeros((256,), dtype=np.float32)
    
    # Loop over each element in the unnormalized histogram.
    # Divide that value by the total and store it in the current index of the
    # normalized histogram array.
    for i in range(hist.shape[0]):
        normalizedHist[i] = hist[i]/total
    
    return normalizedHist
    
def create_cdf(nhist):
    # Create an empty array to store the values into.
    cdf = np.zeros((256,), dtype=np.float32)
    
    # Loop over each element in the normalized histogram.
    # If the index equals 0 set the 0th index in the cdf array to the same value 
    #   as the 0th index in the normalized histogram.
    # Else set the cdf equal to the previous cdf index value plus the current 
    #   index value of the normalized histogram.
    for i in range(nhist.shape[0]):
        if(i==0):
            cdf[i] = nhist[i]
        else:
            cdf[i] = cdf[i-1] + nhist[i]
            
    # Return the CDF
    return cdf
        
def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    ''''''
    
def do_histogram_equalize(image, do_stretching):
    ''''''

def intensity_callback(input_img, do_stretching):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching) 
    return output_img

def main():
    demo = gr.Interface(fn=intensity_callback, 
                        inputs=["image", "checkbox"],
                        outputs=["image"])
    demo.launch() 
    
if __name__ == "__main__":
    main()