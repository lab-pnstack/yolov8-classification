from ultralytics import YOLO
from PIL import Image
import glob
from utils import save_to_file, multi_process_list, process_list,read_txt_content
import re
from multiprocessing import Pool
import timeit
import os
import shutil
from tqdm import tqdm 

model = YOLO('runs/classify/train5/weights/best.pt') 
 
data_dir = 'raw/**/*.png'
good_folder = 'processed/good_images'
bad_folder = 'processed/bad_images'
destination_folder = 'processed'

# Create the folders if they don't exist
if not os.path.exists(good_folder):
    os.makedirs(good_folder)
if not os.path.exists(bad_folder):
    os.makedirs(bad_folder)

# Function to get the top class for an image
def get_top_class(image_path):
    results = model(image_path,verbose=False)
    top_result = results[0]
    probs = top_result.probs
    class_index = probs.top1
    class_name = top_result.names[class_index]
    return class_name

def handle_file(image_path):
    # Get the top predicted class for the image
    top_class = get_top_class(image_path)

    # Move the image to the appropriate folder based on the top class
    if top_class == 'good':
        shutil.move(image_path, good_folder)
        print("Moved: ", image_path)
        pass

    else:
        # shutil.move(image_path, bad_folder)
        pass


# main
def main():
  image = glob.glob(data_dir)  
  process_list(image, handle_file)

if __name__ == "__main__":
  print("Start")
  print(timeit.timeit(main, number=1))
  print("End")
