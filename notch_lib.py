# Custom lib that provides functions for different notch structures
# Created by: Daniel Gore

# Libs
import numpy as np

# Generates a matrix of ideal notches to be element-wise multiplied by an FFT
def notch_matrix(notches_x, notches_y, radii, dims):
    # Matrix params
    width, height = dims
    notch_mat = np.ones((width, height))
    radii = np.array(radii) / 2

    # Create a grid of indices
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y, indexing = 'ij')

    # Apply the notch condition for each (circular) notch
    for center_x, center_y, radius in zip(notches_x, notches_y, radii):
        distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
        notch_mat[distance_sq <= radius ** 2] = 0

    return notch_mat

# Generates a matrix of Gaussian notches to be element-wise multiplied by an FFT
def gaussian_notch_matrix(notches_x, notches_y, radii, dims, sigma):

    # Matrix params
    width, height = dims
    notch_mat = np.ones((width, height))
    radii = np.array(radii) / 2
    
    # Create a grid of indices
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y, indexing = 'ij')

    # Apply the Gaussian notch condition
    for center_x, center_y, radius in zip(notches_x, notches_y, radii):
        
        distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
        gaussian_filter = np.exp(-distance_sq / (2 * sigma ** 2))
        
        # Apply the filter
        mask = distance_sq <= radius ** 2
        notch_mat[mask] = gaussian_filter[mask]

    return notch_mat

# Generates a matrix of notches that are either ideal or Gaussian with added design parameters from the user
def designed_notch_matrix(notches_x, notches_y, radii, dims, minV, maxV, stdev):

    if stdev is not None: # If the filter is Gaussian
        notches = gaussian_notch_matrix(notches_x, notches_y, radii, dims, stdev)
    else:
        notches = notch_matrix(notches_x, notches_y, radii, dims)

        width, height = dims

        for i in range(0, width):
            for j in range(0, height):

                if notches[i][j] < minV:
                    notches[i][j] = minV

                if notches[i][j] > maxV:
                    notches[i][j] = maxV

    return notches
