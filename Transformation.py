import numpy as np
from PIL import Image
import math
import tensorflow as ts
from scipy import ndimage

import tensorflow as ts

          
#angle=int(input("Enter angle :- "))                # User input of angle of rotation

def Rotation(image, angle):
# relevant varibles
    angle =math.radians(angle)     #converting degrees to radians
    cosine =math.cos(angle)
    sine =math.sin(angle)
    height =image.shape[0]                                  
    width =image.shape[1]                                    
# New image height and weight 
    new_height  = round(abs(image.shape[0]*cosine)+abs(image.shape[1]*sine))
    new_width  = round(abs(image.shape[1]*cosine)+abs(image.shape[0]*sine))

# dimensions variable of new_height and new _widnth
    output=np.zeros((new_height,new_width,image.shape[2]))

# define original image centre
    centre_height   = round(((image.shape[0])/2))    
    centre_width    = round(((image.shape[1])/2))    
    
# define new centre
    new_centre_height= round(((new_height)/2))        #with respect to the new image
    new_centre_width= round(((new_width)/2))          #with respect to the new image

    for i in range(height):
        for j in range(width):
        #co-ordinates of pixel with respect to the centre of original image
            y=image.shape[0]-i-centre_height                   
            x=image.shape[1]-j-centre_width                      

        #co-ordinate of pixel with respect to the rotated image
            new_y=round(-x*sine+y*cosine)
            new_x=round(x*cosine+y*sine)

            new_y=new_centre_height-new_y
            new_x=new_centre_width-new_x

        # adding if check to prevent any errors in the processing
            if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x>=0 and new_y>=0:
                output[new_y,new_x,:]=image[i,j,:]   #writing the pixels to the new destination in the output image

    img=Image.fromarray((output).astype(np.uint8))
    return img                      
                                           
    

def shear(image, angle):

    # Define the most occuring variables
    angle=math.radians(angle)                              
    cosine=math.cos(angle)
    sine=math.sin(angle)
    
    height=image.shape[0]                                   
    width=image.shape[1]                                   

    new_height  = round(abs(image.shape[0]*cosine)+abs(image.shape[1]*sine))
    new_width  = round(abs(image.shape[1]*cosine)+abs(image.shape[0]*sine))

    output=np.zeros((new_height,new_width,image.shape[2]))
    image_copy=output.copy()

    centre_height   = round(((image.shape[0])/2))    
    centre_width    = round(((image.shape[1])/2))   


    new_cheight = round(((new_height)/2))        
    new_cwidth = round(((new_width)/2))          

    for i in range(height):
        for j in range(width):
            #co-ordinates of pixel with respect to the centre of original image
            y=image.shape[0]-i-centre_height                   
            x=image.shape[1]-j-centre_width 

            #Applying shear Transformation                     
            tangent=math.tan(angle/2)
            new_x=round(x-y*tangent)
            new_y=y
           

            new_y=new_cheight-new_y
            new_x=new_cwidth-new_x
            
            output[new_y,new_x,:]= image[i,j,:]     

                     #translating new pixel destination in the output image

    
    img=Image.fromarray((output).astype(np.uint8))                       # converting array to image
    return img  

if __name__ == '__main__':
    image = np.array(Image.open("cat.jpg"))   

    #QUESTION 1(b)
   '''
    
    cat30 = Rotation(image, 30)
    cat30.save('output/cat30.jpg', 'JPEG')
    cat60 = Rotation(image, 60)
    cat60.save('output/cat60.jpg', 'JPEG')
    cat120 = Rotation(image, 120)
    cat120.save('output/cat120.jpg', 'JPEG')
 =
    shear10 = shear(image, 10)
    shear10.save('output/shear10.jpg', 'JPEG')
    shear40 = shear(image, 40)
    shear40.save('output/shear40.jpg', 'JPEG')
    shear60 = shear(image, 60)
    shear60.save('output/shear60.jpg', 'JPEG')
    
    '''
    ##1(c)
    #(i)
    catneg20= Rotation(image, -20)
    image2 = np.array(catneg60)   
    shear10 = shear(image2, 10)
    shear10.save('output/rotshear.jpg', 'JPEG')
    
    #(ii)
    shear10 = shear(image, 10)
    image3 = np.array(shear10)
    catneg20 =Rotation(image3, -20)
    catneg20.save('output/shearrot.jpg', 'JPEG')
    
    
    
 
    