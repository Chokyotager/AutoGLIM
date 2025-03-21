import os
import shutil

all_images = [x for x in os.listdir("all_images") if x.lower().endswith(".tif")]

os.makedirs("all_results", exist_ok=True)
os.makedirs("images", exist_ok=True)

for image in sorted(all_images):

    print(f"Running {image}")
    shutil.copy("all_images/" + image, "images/AVG_Composite.tif")
    os.system("python3 process.py")

    shutil.move("results", f"all_results/{image}")
