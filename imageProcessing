from skimage import io
import cv2 as cv
import numpy as np
import os
import random
import itertools
import shutil
from multiprocessing import Pool

def load_and_preprocess_image(path):
  # Load the image
  img = io.imread(path)

  # Check if the image has color channels (3 for RGB, 4 for RGBA) and convert to grayscale
  if len(img.shape) > 2 and img.shape[2] in [3, 4]:
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

  # Take out the first two rows of pixels, the first two columns of pixels,
  # the last four rows of pixels, and the last four columns of pixels
  img_cropped = img[2:-4, 2:-4]

  # Resize the image to 97x90
  img_cropped = cv.resize(img_cropped, (97, 90))

  return img_cropped

# folder can be "real", "easy", "medium", "hard"
# suffix can be "Zcut", "CR", "Obl"
def get_file_path(filename, folder, suffix):
  # Check if the name contains .BMP first
  name, ext = os.path.splitext(filename)
  if folder == "real":
    path = "SOCOFing/Real/{0}.BMP".format(name)
    return path if os.path.isfile(path) else None
  elif folder == "easy":
    path = "SOCOFing/Altered/Altered-Easy/{0}_{1}.BMP".format(name, suffix)
    return path if os.path.isfile(path) else None
  elif folder == "medium":
    path = "SOCOFing/Altered/Altered-Medium/{0}_{1}.BMP".format(name, suffix)
    return path if os.path.isfile(path) else None
  elif folder == "hard":
    path = "SOCOFing/Altered/Altered-Hard/{0}_{1}.BMP".format(name, suffix)
    return path if os.path.isfile(path) else None
  return None

def list_files_in_directory(directory_path, filter = ""):
  with os.scandir(directory_path) as entries:
      files = [entry.name for entry in entries if entry.is_file() and filter in entry.name]
  random.shuffle(files)
  return files

def make_folders():
  # List of directories to create
  dirs = ["data/training/false", "data/training/true", "data/testing/false", "data/testing/true"]

  # Iterate through the list and create each directory
  for directory in dirs:
    # If the directory already exists, delete it
    if os.path.exists(directory):
       shutil.rmtree(directory)
    # Create the directory
    os.makedirs(directory)

def random_zoom_rotate(img, zoom_range, rotation_range):
  # Invert the image
  img = 255 - img

  # Select random zoom factor and rotation angle
  zoom = np.random.uniform(zoom_range[0], zoom_range[1])
  angle = np.random.uniform(rotation_range[0], rotation_range[1])
  
  # Get image height and width
  height, width = img.shape[:2]

  # Compute the rotation matrix
  rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), angle, zoom)

  # Perform the rotation and zoom
  img_transformed = cv.warpAffine(img, rotation_matrix, (width, height))

  return img_transformed

def process_combination(args):
  i, combination, length, prefix = args
  progress = (i / length) * 100  # Calculate the percentage progress
  print(f"\rProcessing: {progress:.2f}%", end="")

  key_1 = combination[0][1]
  key_2 = combination[1][1]
  img_1 = load_and_preprocess_image(combination[0][0])
  img_2 = load_and_preprocess_image(combination[1][0])

  # Define your zoom and rotation ranges
  zoom_range = [0.9, 1.1]  # zoom in or out by 10%
  rotation_range = [-10, 10]  # rotate between -10 and +10 degrees

  # Apply random zoom and rotation
  img_1 = random_zoom_rotate(img_1, zoom_range, rotation_range)
  img_2 = random_zoom_rotate(img_2, zoom_range, rotation_range)

  if img_1.shape != img_2.shape:
    return

  final_frame = np.concatenate((img_1, img_2), axis=1)

  if key_1 == key_2:
    destination = "data/" + prefix + "/true/true_" + str(i) + ".png"
    cv.imwrite(destination, final_frame)
  else:
    destination = "data/" + prefix + "/false/false_" + str(i) + ".png"
    cv.imwrite(destination, final_frame)

def make_data(keys, prefix):
  # Extract images into a list
  pool = []

  for key in keys:
    pool.append((get_file_path(key, "real", ""), key))
    for level in ["easy", "medium", "hard"]:
      for variant in ["Zcut", "CR", "Obl"]:
        path = get_file_path(key, level, variant)
        if path is not None:
          pool.append((path, key))

  # Create all combinations of 2 items
  combinations = list(itertools.combinations(pool, 2))
  # Shuffle the combinations
  random.shuffle(combinations)

  with Pool() as p:
    # Use the Pool's map method to run process_combination for each combination
    # Pack all arguments into a tuple
    p.map(process_combination, [(i, combination, len(combinations), prefix) for i, combination in enumerate(combinations)])

def main():
  kSeed = 12345
  random.seed(kSeed)

  # We have 6000 fingers in total
  kTrainingSetSize = 50
  kTestingSetSize = 50

  files = list_files_in_directory("SOCOFing/Real")

  make_folders()

  make_data(files[:kTrainingSetSize], "training")
  make_data(files[kTrainingSetSize : kTrainingSetSize + kTestingSetSize], "testing")

if __name__ == "__main__":
  main()
