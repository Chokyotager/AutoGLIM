import argparse

import cv2
import numpy as np
import os
import shutil

import csv

from tqdm import tqdm

import torch
from scipy import stats, optimize

from filter_contours import filter_contours, filter_contours_centre_of_mass
from compute_beads import filter_beads, filter_beads_centre_of_mass
import compute_metrics
import chromatic_correction

import roifile

import matplotlib.pyplot as plt
import seaborn as sns

from config import *

sns.set_style("whitegrid", {"axes.grid": False})

parser = argparse.ArgumentParser(
    prog="AutoGLIM",
    description="Automated GLIM processing",
    epilog="Version 1.0"
)

parser.add_argument("--input", required=False, type=str, help="Path to the output folder", default="images/AVG_Composite.tif")
parser.add_argument("--beads", required=False, type=str, help="Path to the reference beads folder", default="beads_images")
parser.add_argument("--output", required=False, type=str, help="Path to the output folder", default="results")
parser.add_argument("--reference_beads", required=False, type=str, help="Path to the precomputed beads TSV")
parser.add_argument('--overwrite', action="store_true", help="Overwrite old results if it exists")

args = parser.parse_args()

reference_beads_file = args.reference_beads
input_image = args.input
beads_images_folder = args.beads
results_folder = args.output

print("\nAutoGLIM version 1.0\nLast updated: 21 March 2025 \n(Coded by Hilbert Lam for A/Prof. Lu Lei)")

if os.path.exists(args.output):
    if args.overwrite:
        shutil.rmtree(args.output)

    else:
        raise "Results folder already exists. Specify --overwrite or manually remove the folder."

os.makedirs(args.output, exist_ok=True)

"""
Step 1: Beads computation
"""

reference_beads = f"{results_folder}/reference-beads.tsv"

if reference_beads_file and os.path.isfile(reference_beads_file) and not force_beads_calculation:

    print("\nNOTE: reference-beads.tsv is provided; will use that for chromatic correction calculation!")

    shutil.copy(reference_beads_file, reference_beads)

else:

    print("\nNOTE: will perform beads calculation since reference-beads.tsv is not provided!")

    print("\n=== Beads calibration")

    beads_images = [x for x in sorted(os.listdir(beads_images_folder)) if x.endswith(".tif")]

    if output_beads_calibration:
    
        os.makedirs(f"{results_folder}/beads_calibration", exist_ok=True)

    all_contours = list()

    for beads_image in beads_images:

        beads_image_id = beads_image.replace(".tif", "")

        composite_image = cv2.imreadmulti(beads_images_folder + "/" + beads_image, flags=cv2.IMREAD_UNCHANGED)

        composite_image = np.array(composite_image[1])
        image = np.moveaxis(composite_image, 0, -1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_beads_image = image

        channels = cv2.split(image)

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(128, 128))
        channels = [clahe.apply(x) for x in channels]

        binarised_channels = [x > np.percentile(x.flatten(), beads_percentile) for x in channels]

        image = np.moveaxis(np.stack(binarised_channels), 0, -1).astype(np.uint8) * 255

        if centring_method == "centroid":

            correct_contours = filter_beads(image, beads_particle_size_threshold, minimum_colour_channel_pixels_beads, minimum_circularity)

        elif centring_method == "com":

            correct_contours = filter_beads_centre_of_mass(original_beads_image, image, beads_particle_size_threshold, minimum_colour_channel_pixels_beads, minimum_circularity)

        mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [x["contour"] for x in correct_contours], color=1)

        bgst_norm = (image > 0).astype(np.uint8)

        #norm_bgst = cv2.normalize(bgst, None, 0, 255, cv2.NORM_MINMAX)

        if output_beads_calibration:

            cv2.imwrite(f"{results_folder}/beads_calibration/filtered_beads_{beads_image_id}.png", bgst_norm * mask[:, :, np.newaxis] * 255)

        print(f"Obtained {len(correct_contours)} beads from {beads_image}")
        all_contours.extend(correct_contours)

    coordinates = list()

    for contour in all_contours:

        coordinates.append(contour[centring_method][0] + contour[centring_method][1] + contour[centring_method][2])

    writable = [["Xm_R", "Ym_R", "Xm_G", "Ym_G",  "Xm_B", "Ym_B"]] + coordinates
    open(reference_beads, "w+").write("\n".join(["\t".join([str(y) for y in x]) for x in writable]))

    print(f"A total of {len(all_contours)} beads obtained for calibration.")

"""
Step 2: Cell segmentation
"""

composite_image = cv2.imreadmulti(input_image)

composite_image = np.array(composite_image[1])
bgst = np.moveaxis(composite_image, 0, -1)

channels = cv2.split(bgst)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
channels = [clahe.apply(cv2.equalizeHist(x)) for x in channels]
channels = [cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX) for x in channels]

image = np.moveaxis(np.stack(channels), 0, -1)
original_shape = image.shape

# Pad image to nearest 16
new_height = (image.shape[0] + 15) // 16 * 16
new_width = (image.shape[1] + 15) // 16 * 16

pad_height = new_height - image.shape[0]
pad_width = new_width - image.shape[1]

image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=0)

device = torch.device("cpu")
model = torch.load("model_final.pt", map_location=device, weights_only=False)

model.to(device)
model.eval()

input_image_tensor = torch.from_numpy(image).swapaxes(0, -1).unsqueeze(0)
output_segmentation = model(input_image_tensor.to(device) / 255).sigmoid().detach().cpu().numpy()[0]
mask = (output_segmentation * 255).swapaxes(0, 1).astype(np.uint8)

mask = mask[:original_shape[0], :original_shape[1]]
mask = (cv2.medianBlur(mask, 31) > cell_segmentation_cutoff).astype(np.uint8)

kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = np.uint8)
mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

minimum_segmentation_area = int(minimum_segmentation_ratio * original_shape[0] * original_shape[1])
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
contours = [[y[0] for y in x.tolist()] for x in contours if cv2.contourArea(x) > minimum_segmentation_area]

for contour in contours:

    contour = np.array(contour).reshape((-1, 1, 2))
    image = cv2.polylines(image, [contour], True, (0, 255, 255), 5)

cv2.imwrite(f"{results_folder}/cell_segmentation.png", image)

"""
Step 3: Automated background subtraction
"""

print("\n=== Automated background subtraction")

# Automated contrasting
composite_image = cv2.imreadmulti(input_image, flags=cv2.IMREAD_UNCHANGED)
composite_image = np.array(composite_image[1])
image = np.moveaxis(composite_image, 0, -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

channels = cv2.split(image)

clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid)
channels = [(clahe.apply(x)) for x in channels]

def threshold_calculate_ministacks (masked_channels, thresholds):

    all_binarised = list()

    # Flatten the masked values into a 1D array
    for channel, percentile_threshold in zip(masked_channels, thresholds):

        flattened = channel[mask == 1].flatten()

        percentile_value = np.percentile(flattened, percentile_threshold)

        binarised = (channel > percentile_value).astype(np.uint8)
        all_binarised.append(binarised)

    stacked = np.stack(all_binarised)
    stacked = np.moveaxis(stacked, 0, -1)

    contours = filter_contours(stacked, particle_size_threshold=particle_size_threshold, minimum_channel_contour_area=minimum_colour_channel_pixels_contour)

    return len(contours)

combined = np.zeros_like(image)

for contour in tqdm(contours):

    mask = np.zeros([original_shape[0], original_shape[1]])

    contour = np.array(contour).reshape((-1, 1, 2))
    mask = cv2.fillPoly(mask, [contour], (1, 1, 1))[:original_shape[0], :original_shape[1]]
    masked_channels = [mask * x for x in channels]

    def eval_function(params):

        x, y, z = params
        num_contours = threshold_calculate_ministacks(masked_channels, [x, y, z])

        return -num_contours

    result = optimize.dual_annealing(eval_function, percentile_bounds, maxiter=max_opt, maxfun=max_opt, seed=0)
    percentile_thresholds = result.x.tolist()

    percentile_thresholds = [95, 95, 95]

    all_binarised = list()

    # Flatten the masked values into a 1D array
    for channel, percentile_cutoff in zip(masked_channels, percentile_thresholds):

        flattened = channel[mask == 1].flatten()

        percentile_value = np.percentile(flattened, percentile_cutoff)

        binarised = (channel > percentile_value).astype(np.uint8)
        all_binarised.append(binarised)

    stacked = np.stack(all_binarised)
    stacked = np.moveaxis(stacked, 0, -1)

    combined += stacked

combined = combined.astype(np.uint8)
cv2.imwrite(f"{results_folder}/bgst.png", combined * 255)

"""
Step 4: Contours
"""

bgst_image = combined

if output_roi_images:
    os.makedirs(f"{results_folder}/selected_ROIs", exist_ok=True)

# Select and filter contours
# Calculate by centre of mass (with BGST correction)
#correct_contours = filter_contours_centre_of_mass(image, bgst_image, particle_size_threshold=particle_size_threshold, minimum_channel_contour_area=minimum_channel_contour_area)

if centring_method == "centroid":
    correct_contours = filter_contours(bgst_image, particle_size_threshold=particle_size_threshold, minimum_channel_contour_area=minimum_colour_channel_pixels_contour)

elif centring_method == "com":
    correct_contours = filter_contours_centre_of_mass(image, bgst_image, particle_size_threshold=particle_size_threshold, minimum_channel_contour_area=minimum_colour_channel_pixels_contour)

coordinates = list()

current_contour = 0
cropped_rois = list()

ijrois = list()

for correct_contour in correct_contours:

    contour = correct_contour["contour"]
    area = cv2.contourArea(contour)

    com_data = correct_contour[centring_method]
    current_data = [area] + com_data[2] + com_data[1] + com_data[0]

    coordinates.append([x for x in current_data])

    current_roi = roifile.ImagejRoi.frompoints(contour + 0.5)
    current_roi.roitype = roifile.ROI_TYPE.POLYGON
    current_roi.options |= roifile.ROI_OPTIONS.SHOW_LABELS
    current_roi.name = f"ROI_{current_contour + 1}"

    ijrois.append(current_roi)

    if output_roi_images:

        # Compute bounding box
        x1 = min([x[0] for x in contour])
        y1 = min([x[1] for x in contour])

        x2 = max([x[0] for x in contour])
        y2 = max([x[1] for x in contour])

        mask = np.zeros_like(bgst_image[:,:,0], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [contour], color=1)

        cropped_image = bgst_image * mask[:, :, np.newaxis] * 255
        cropped_image = cropped_image[y1:y2, x1:x2]

        cv2.imwrite(f"{results_folder}/selected_ROIs/ROI_{current_contour + 1}.png", cropped_image)
        cropped_rois.append(cropped_image)

    current_contour += 1

roifile.roiwrite(f"{results_folder}/selected_ROIs.ijroi.zip", ijrois)

# Output coordinates.tsv
writable = [["Area_RGB", "Xm_R", "Ym_R", "Xm_G", "Ym_G",  "Xm_B", "Ym_B"]] + coordinates
open(f"{results_folder}/coordinates.tsv", "w+").write("\n".join(["\t".join([str(y) for y in x]) for x in writable]))

print("\n=== Contours")
print("Total ROIs identified:", len(correct_contours))
mask = np.zeros_like(bgst[:,:,0], dtype=np.uint8)
mask = cv2.fillPoly(mask, [x["contour"] for x in correct_contours], color=1)

cv2.imwrite(f"{results_folder}/selected_contours.png", bgst_image * 255 * mask[:, :, np.newaxis])

"""
Step 5: Chromatic aberration correction
"""

# Perform shift correction
beads_data = list(csv.reader(open(reference_beads), delimiter="\t"))[1:]

beads_x_r = np.array([float(x[0]) for x in beads_data])
beads_y_r = np.array([float(x[1]) for x in beads_data])

beads_x_g = np.array([float(x[2]) for x in beads_data])
beads_y_g = np.array([float(x[3]) for x in beads_data])

beads_x_b = np.array([float(x[4]) for x in beads_data])
beads_y_b = np.array([float(x[5]) for x in beads_data])

# Calculate chromatic shift
dgrx_coeff, dgry_coeff, dbrx_coeff, dbry_coeff = chromatic_correction.calculate_shift(beads_x_r, beads_y_r, beads_x_g, beads_y_g, beads_x_b, beads_y_b)

# Shifted by 1 due to area
raw_coordinates = [x[1:] for x in coordinates]

x_r = np.array([float(x[0]) for x in raw_coordinates])
y_r = np.array([float(x[1]) for x in raw_coordinates])

x_g = np.array([float(x[2]) for x in raw_coordinates])
y_g = np.array([float(x[3]) for x in raw_coordinates])

x_b = np.array([float(x[4]) for x in raw_coordinates])
y_b = np.array([float(x[5]) for x in raw_coordinates])

x_g2r_res, y_g2r_res, x_b2r_res, y_b2r_res = chromatic_correction.correct_shift(x_g, y_g, x_b, y_b, dgrx_coeff, dgry_coeff, dbrx_coeff, dbry_coeff)
x_error_g2r_corrected, y_error_g2r_corrected, x_error_b2r_corrected, y_error_b2r_corrected = chromatic_correction.calculate_error(x_r, y_r, x_g, y_g, x_b, y_b, dgrx_coeff, dgry_coeff, dbrx_coeff, dbry_coeff)

fig = plt.figure(figsize=(5, 5))

print("\n=== Chromatic shift corrections")
print(f"Mean green channel x-axis error: {np.mean(x_error_g2r_corrected):.3f} ± {np.std(x_error_g2r_corrected):.3f} px (S.D.)")
print(f"Mean green channel y-axis error: {np.mean(y_error_g2r_corrected):.3f} ± {np.std(y_error_g2r_corrected):.3f} px (S.D.)")
print(f"Mean blue channel x-axis error: {np.mean(x_error_b2r_corrected):.3f} ± {np.std(x_error_b2r_corrected):.3f} px (S.D.)")
print(f"Mean blue channel y-axis error: {np.mean(y_error_b2r_corrected):.3f} ± {np.std(y_error_b2r_corrected):.3f} px (S.D.)")

plt.scatter(x_error_g2r_corrected, y_error_g2r_corrected, s=5, c="green")
plt.scatter(x_error_b2r_corrected, y_error_b2r_corrected, s=5, c="blue")

plt.title("Chromatic shift corrections")
plt.axvline(0, color="black", linewidth=1)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Corrections on x-coordinate / px")
plt.ylabel("Corrections on y-coordinate / px")

plt.tight_layout()

ax = plt.gca()

for spine in ax.spines.values():
    spine.set_edgecolor("black")

plt.savefig(f"{results_folder}/corrections.png", dpi=300)
plt.clf()

# Edit coordinates

all_shifted_coordinates = list()
for i in range(len(coordinates)):

    current_coordinates = coordinates[i]

    shifted_coordinates = current_coordinates[1:3] + [x_g2r_res[i], y_g2r_res[i], x_b2r_res[i], y_b2r_res[i]]
    all_shifted_coordinates.append(shifted_coordinates)

writable = [["Xm_R", "Ym_R", "Xm_G_shifted", "Ym_G_shifted",  "Xm_B_shifted", "Ym_B_shifted"]] + all_shifted_coordinates
open(f"{results_folder}/shifted_coordinates.tsv", "w+").write("\n".join(["\t".join([str(y) for y in x]) for x in writable]))

"""
Step 6: LQ and metrics calculation
"""

raw_metrics = list()
filtered_metrics = list()
filtered_cropped_rois = list()
filtered_ijrois = list()

for i in range(len(all_shifted_coordinates)):

    current_coordinates = all_shifted_coordinates[i]

    lq, dx, d1, abs_tan_a, abs_tan_b = compute_metrics.calculate_localisation_quotient(*current_coordinates)

    raw_metrics.append(current_coordinates + [dx, d1, abs_tan_a, abs_tan_b, lq])

    condition_1 = abs(d1) >= minimum_intercentroid_distance
    condition_2 = abs_tan_a <= maximum_tangent_alpha or abs_tan_b <= maximum_tangent_beta

    if all([condition_1, condition_2]):
        filtered_metrics.append(current_coordinates + [dx, d1, abs_tan_a, abs_tan_b, lq])
        filtered_ijrois.append(ijrois[i])

        if output_roi_images:
            filtered_cropped_rois.append(cropped_rois[i])

if output_roi_images:
    os.makedirs(f"{results_folder}/filtered_ROIs", exist_ok=True)

    for i in range(len(filtered_cropped_rois)):
        filtered_ijrois[i].name = f"ROI_{i + 1}"
        cv2.imwrite(f"{results_folder}/filtered_ROIs/ROI_{i + 1}.png", filtered_cropped_rois[i])

roifile.roiwrite(f"{results_folder}/filtered_ROIs.ijroi.zip", filtered_ijrois)

writable = [["Xm_R", "Ym_R", "Xm_G_shifted", "Ym_G_shifted",  "Xm_B_shifted", "Ym_B_shifted", "dx", "d1", "abs(tan(alpha))", "abs(tan(beta))", "LQ"]] + raw_metrics
open(f"{results_folder}/unfiltered_metrics.tsv", "w+").write("\n".join(["\t".join([str(y) for y in x]) for x in writable]))

writable = [["Xm_R", "Ym_R", "Xm_G_shifted", "Ym_G_shifted",  "Xm_B_shifted", "Ym_B_shifted", "dx", "d1", "abs(tan(alpha))", "abs(tan(beta))", "LQ"]] + filtered_metrics
open(f"{results_folder}/filtered_metrics.tsv", "w+").write("\n".join(["\t".join([str(y) for y in x]) for x in writable]))

lqs = [x[-1] for x in filtered_metrics]

print("\n=== Metrics")
print(f"Final ROIs: {len(filtered_metrics)}")
print(f"Average LQ: {np.mean(lqs):.5f} ± {stats.sem(lqs):.5f} (S.E.M.)")

"""
Step 7: Plot
"""

fig = plt.figure(figsize=(3, 3))

plt.grid(axis="y")

sns.boxplot(lqs)
plt.ylabel("LQ")

plt.tight_layout()

ax = plt.gca()

ax.axes.get_xaxis().set_ticks(list())

for spine in ax.spines.values():
    spine.set_edgecolor("black")

plt.savefig(f"{results_folder}/LQ.png", dpi=300)
plt.clf()

fig = plt.figure(figsize=(5, 5))

plt.ylabel("Number of ROIs")
plt.xlabel("LQ")

sns.histplot(lqs, kde=True, edgecolor="black", linewidth=1, color="#003166")

plt.tight_layout()

ax = plt.gca()

for spine in ax.spines.values():
    spine.set_edgecolor("black")

plt.savefig(f"{results_folder}/LQ_histogram.png", dpi=300)