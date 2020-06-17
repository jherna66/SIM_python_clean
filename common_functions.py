import numpy as np

def fft(image):
    if (len(image.shape) > 2):
        spectrum = np.fft.fftshift(np.fft.fft2(image),axes=(1,2))
    else:
        spectrum = np.fft.fftshift(np.fft.fft2(image))
    return spectrum

def ifft(spectrum):
    if (len(spectrum.shape) > 2):
        image = np.fft.ifft2(np.fft.ifftshift(spectrum,axes=(1,2)))
    else:
        image = np.fft.ifft2(np.fft.ifftshift(spectrum))
    return image

def max_value_index(matrix):
    '''Returns a tuple which are the index of the max of the matrix'''
    
    # flatten matrix
    maxElement = np.amax(np.ravel(matrix))
    x, y = np.where(matrix == maxElement)
    return x[0], y[0]

def pad_image(image, target_width, pad_value):
    size = image.shape[0]
    pad_size = (target_width - size)//2
    pad_image = np.pad(image, pad_size, 'constant', constant_values=pad_value)
    return pad_image

def apodization(size, threshold, fvector, angles):
    '''
    in:
    size - size of the image
    threshold - maximum frequency supported by the otf
    fvector - frequency of SI
    angles - angles of SI
    '''
    ADF_s = np.repeat(np.zeros((size,size),dtype=np.complex128)[np.newaxis], angles.size * 3, axis=0)
    ADF = np.zeros((size,size),dtype=np.complex128)
    n = np.arange(0, size, 1)
    for t in range(angles.size * 3):
        # Create blank canvas for apod_fun
        
        # Create the vector for angles
        angle_arr = np.repeat(angles, 3, axis=0)
        xshift = fvector * np.cos(angle_arr[t])
        yshift = fvector * np.sin(angle_arr[t])
        if (t%3 == 0):
            x, y = np.meshgrid(np.abs(n - size//2),np.abs(n - size//2))
            r = np.sqrt(np.power(x,2) + np.power(y,2))
            #ADF[r < threshold] = 1
            ADF_s[t,r < threshold] = 1
        elif (t%3 == 1):
            # Shift to the right
            x, y = np.meshgrid(np.abs(n - size//2 + xshift),np.abs(n - size//2 + yshift))
            r = np.sqrt(np.power(x,2) + np.power(y,2))
            #ADF[r < threshold] = 1
            ADF_s[t,r < threshold] = 1
        elif (t%3 == 2):
            # Shift to the left
            x, y = np.meshgrid(np.abs(n - size//2 - xshift),np.abs(n - size//2 - yshift))
            r = np.sqrt(np.power(x,2) + np.power(y,2))
            #ADF[r < threshold] = 1
            ADF_s[t,r < threshold] = 1
        
    return ADF_s

