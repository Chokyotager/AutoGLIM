import cv2
import numpy as np

from scipy.ndimage import center_of_mass

def filter_contours (bgst_image, particle_size_threshold, minimum_channel_contour_area):

    bgst = bgst_image

    image_dims = [bgst.shape[0], bgst.shape[1]]

    """"
    Original: GLIM_1_Golgi particle analysis.ijm
    """

    # Thresholding
    binarised = (bgst > 0)

    # OR operation across three channels
    # to create a single greyscale image
    greyscaled = np.any(binarised, axis=-1).astype(np.uint8)

    # Find all contours of defined size
    contours, _ = cv2.findContours(greyscaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered_contours = [x for x in contours if particle_size_threshold[0] < cv2.contourArea(x) < particle_size_threshold[1]]

    # Sort contours by size
    filtered_contours = sorted(filtered_contours, key=lambda x: cv2.contourArea(x), reverse=True)

    """"
    Original: GLIM_2_Golgi ROI inspection.ijm
    Modified: automated contour filtering

    Original: GLIM_3_Golgi output 3 channel data.ijm
    """

    # Check for contiguity in contours for each channel
    # Remove all discontiguous contours

    correct_contours = list()

    for filtered_contour in filtered_contours:

        # Polygons are traced
        contour = filtered_contour[:,0]

        # Remove contours that touch the edge
        contour_x = [x[0] for x in contour]
        contour_y = [x[1] for x in contour]

        overlap_x = any([x for x in contour_x if x >= image_dims[1] - 1 or x <= 1])
        overlap_y = any([y for y in contour_y if y >= image_dims[0] - 1 or y <= 1])

        if overlap_x or overlap_y:
            continue

        mask = np.zeros_like(bgst[:,:,0], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [contour], color=1)

        correct_channels = True

        centroids = list()

        for i in range(3):
            
            # Extract the ROI
            current_channel = bgst[:,:,i] * mask

            # Count contours in single channel
            channel_contours, _ = cv2.findContours(current_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            number_channel_contours = len(channel_contours)

            if number_channel_contours != 1:
                correct_channels = False
                break

            if cv2.contourArea(channel_contours[0]) < minimum_channel_contour_area:
                correct_channels = False
                break

            centroid = channel_contours[0].mean(axis=0)[0]
            centroids.append(centroid.tolist())

        if correct_channels:

            correct_contours.append({"centroid":  centroids, "contour": contour})

    return correct_contours

# Calculates centre of mass, not centroid
def filter_contours_centre_of_mass (original_image, bgst_image, particle_size_threshold, minimum_channel_contour_area):

    bgst = bgst_image

    image_dims = [bgst.shape[0], bgst.shape[1]]

    """"
    Original: GLIM_1_Golgi particle analysis.ijm
    """

    # Thresholding
    binarised = (bgst > 0)

    # OR operation across three channels
    # to create a single greyscale image
    greyscaled = np.any(binarised, axis=-1).astype(np.uint8)

    # Find all contours of defined size
    contours, _ = cv2.findContours(greyscaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered_contours = [x for x in contours if particle_size_threshold[0] < cv2.contourArea(x) < particle_size_threshold[1]]

    # Sort contours by size
    filtered_contours = sorted(filtered_contours, key=lambda x: cv2.contourArea(x), reverse=True)

    """"
    Original: GLIM_2_Golgi ROI inspection.ijm
    Modified: automated contour filtering

    Original: GLIM_3_Golgi output 3 channel data.ijm
    """

    # Check for contiguity in contours for each channel
    # Remove all discontiguous contours

    correct_contours = list()

    for filtered_contour in filtered_contours:

        # Polygons are traced
        contour = filtered_contour[:,0]

        # Remove contours that touch the edge
        contour_x = [x[0] for x in contour]
        contour_y = [x[1] for x in contour]

        overlap_x = any([x for x in contour_x if x >= image_dims[1] - 1 or x <= 1])
        overlap_y = any([y for y in contour_y if y >= image_dims[0] - 1 or y <= 1])

        if overlap_x or overlap_y:
            continue

        mask = np.zeros_like(bgst[:,:,0], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [contour], color=1)

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = np.uint8)
        corona_mask = np.clip(cv2.dilate(mask, kernel, iterations=1) - mask, a_min=0, a_max=1)

        correct_channels = True

        coms = list()

        for i in range(3):
            
            # Extract the ROI
            current_channel = bgst[:,:,i] * mask

            # Count contours in single channel
            channel_contours, _ = cv2.findContours(current_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            number_channel_contours = len(channel_contours)

            if number_channel_contours != 1:
                correct_channels = False
                break

            if cv2.contourArea(channel_contours[0]) < minimum_channel_contour_area:
                correct_channels = False
                break

            channel_contour = np.array(channel_contours[0])[:,0]

            channel_mask = np.zeros_like(original_image[:,:,0], dtype=np.uint8)
            channel_mask = cv2.fillPoly(channel_mask, [channel_contour], color=1)

            background_channel_intensities = original_image[:,:,i][corona_mask == 1]

            background_intensity_mean = np.mean(background_channel_intensities)
            background_intensity_stddev = np.std(background_channel_intensities) + 1e-8

            subtracted_roi = np.clip((original_image[:,:,i] - background_intensity_mean) * channel_mask / background_intensity_stddev, a_min=0, a_max=None)

            if np.sum(subtracted_roi) == 0:
                correct_channels = False
                break

            com = center_of_mass(subtracted_roi)
            coms.append([com[1], com[0]])

        if correct_channels:

            correct_contours.append({"com":  coms, "contour": contour})

    return correct_contours