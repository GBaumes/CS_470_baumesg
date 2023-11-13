# Assignment 2 Code
# Written by: Gregory Baumes

import cv2
import gradio as gr
import numpy as np

def read_kernel_file(filepath):
    # Open the file
    contents = []
    with open(filepath, "r") as file:
        # Grab the first line and store it in contents
        contents = file.readline()
        # Close the file
        file.close()

    # Split the contents into tokens by spaces.    
    tokens = contents.split()
    
    # Grab the row count and the column count and convert them to ints.
    rowCnt = int(tokens[0])
    colCnt = int(tokens[1])
    
    # Create empty numpy array shape(rowCnt, colCnt).
    kernel = np.zeros((rowCnt, colCnt))
    
    # Set index equal to 2 
    index = 2
    # Loop over kernel
    for row in range(rowCnt):
        for col in range(colCnt):
            # Set value of current kernel element
            kernel[row, col] = float(tokens[index])
            # Increment index by 1
            index += 1
            
    return kernel
        
        
def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    # Cast image and kernel to float64
    image = image.astype("float64")
    kernel = kernel.astype("float64")
    
    # Rotate the kernel 180 degrees
    kernel = cv2.flip(kernel, -1)
    
    # Create a padded image
    paddedWidth = kernel.shape[0] // 2
    paddedHeight = kernel.shape[1] // 2
    paddedImage = cv2.copyMakeBorder(image,paddedWidth, paddedWidth, paddedHeight, paddedHeight, borderType=cv2.BORDER_CONSTANT, value=0)
    
    # Create a floating-point numpy array to hold output image
    outputImage = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
    
    # Loop over each pixel and change the value to the new value.
    for row in range(outputImage.shape[0]):
        for col in range(outputImage.shape[1]):
            subimage = paddedImage[row:(row+kernel.shape[0]), col:(col+kernel.shape[1])]
            filtervals = subimage * kernel
            value = np.sum(filtervals)
            outputImage[row,col] = value
    
    # Check if convert_uint8 is set to True and use convertScaleAbs to chagne the outputImage.
    if convert_uint8==True:
        outputImage = cv2.convertScaleAbs(outputImage, alpha=alpha, beta=beta)
        
    return outputImage
        
def filtering_callback(input_img, filter_file, alpha_val, beta_val): 
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) 
    kernel = read_kernel_file(filter_file.name) 
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val)     
    return output_img 
 
def main(): 
    demo = gr.Interface(fn=filtering_callback,  
                        inputs=["image",  
                                "file",  
                                gr.Number(value=0.125),  
                                gr.Number(value=127)], 
                        outputs=["image"]) 
    demo.launch()
    
# Later, at the bottom 
if __name__ == "__main__":  
    main()
    