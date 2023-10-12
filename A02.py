# Assignment 2 Code
# Written by: Gregory Baumes

import cv2
import gradio as gr


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