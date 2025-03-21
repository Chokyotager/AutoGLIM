# Beads calibration
beads_percentile = 99

# Cell segmentation and automated background subtraction
cell_segmentation_cutoff = 0.01 # Control for AI model, higher is stricter
minimum_segmentation_ratio = 0.01 # Minimum area for AI segmentation
dilation_iterations = 24 # Expanding the segmentation
clahe_clip_limit = 5.0 # Default - 5.0
clahe_grid = (128, 128) # Default - 128, 128
percentile_bounds = [(85, 99), (85, 99), (85, 99)] # Bounds to optimise
max_opt = 250 # Higher value indicates increased exhaustiveness

# Contouring
particle_size_threshold = [20, 1000] # Minimum and maximum particle size for Golgi ministack in pixels
minimum_colour_channel_pixels_contour = 9 # Minimum number of pixels per channel in contour
centring_method = "com" # Either centroid or com (centre of mass)

# Chromatic aberration correction 
# Notice: beads are always calculated using centroid
minimum_colour_channel_pixels_beads = 1 # Minimum number of pixels per channel in beads contour
beads_particle_size_threshold = [10, 120] # Minimum and maximum particle size for beads
minimum_circularity = 0.8 # Minimum circularity for beads

# LQ calculation
minimum_intercentroid_distance = 0.9
maximum_tangent_alpha = 0.3
maximum_tangent_beta = 0.3

# Set below to True if you want to force beads calculation
force_beads_calculation = False

# Output control
output_roi_images = True # Gallery
output_beads_calibration = True
