import cv2, math
import numpy as np
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import shutil
import os

def load_image(path_name):
  image = cv2.imread(path_name)
  return image

def canny_edge(image, minVal=40, maxVal=80, L2 = True):
  edges = cv2.Canny(image, minVal, maxVal, 3, L2gradient=L2)
  return edges

def get_points_for_radius(radius, num_points = 500):
  all_points = []
  for t in range(num_points):
    radians = 2*math.pi*t / num_points
    x = int(radius * math.cos(radians))
    y = int(radius * math.sin(radians))
    all_points.append((x,y))

  return all_points

def get_points_for_all_radius(all_radiuses):
  points = {}
  for each_radius in all_radiuses:
    points_for_each_radius = get_points_for_radius(each_radius)
    points[each_radius] = points_for_each_radius

  return points

def checkValid(a,b,rows,cols):
  if a>=0 and a<cols and b>=0 and b<rows:
    return True
  else:
    return False

def fill_accumulator_map(edges_image, rMin, rMax, step = 1):
  all_radiuses = np.arange(rMin, rMax+1, step)
  points_map = get_points_for_all_radius(all_radiuses)

  folder = tempfile.mkdtemp()
  edges_image_folder = os.path.join(folder, 'edges_image')
  dump(edges_image, edges_image_folder)
  edges_image = load(edges_image_folder, mmap_mode='r')

  points_map_folder = os.path.join(folder, 'points_map')
  dump(points_map, points_map_folder)
  points_map = load(points_map_folder, mmap_mode='r')

  results = Parallel(n_jobs=-1)(delayed(fill_for_each_radius)(edges_image, points_map, r) for r in all_radiuses)

  try:
    shutil.rmtree(folder)
  except:
    print("Failed to delete: " + folder)

  return results

def fill_for_each_radius(edges_image , points_map , each_radius):
  accumulator_map = {}
  points_for_each_radius = points_map[each_radius]
  for i in range(edges_image.shape[0]):
    for j in range(edges_image.shape[1]):
      pixel_value = edges_image[i][j]
      if pixel_value > 0:
        for each_point in points_for_each_radius:
          dx, dy = each_point
          a = j - dx
          b = i - dy
          if checkValid(a,b,edges_image.shape[0],edges_image.shape[1]):
            key = (a,b,each_radius)
            if key not in accumulator_map:
              accumulator_map[key] = 1
            else:
              accumulator_map[key] += 1

  return accumulator_map

def get_circles_with_threshold(results, threshold = 0.2, num_points = 500):
  valid_circles = []
  for each_map in results:
    for k, v in each_map.items():
      if ( v / num_points >= threshold ):
        valid_circles.append(k)
  return valid_circles

if __name__ == "__main__":
  img = load_image('Q1.jpeg')
  grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imwrite('Q1-Output/gray.jpeg', grayscale_image)
  blurred = cv2.GaussianBlur(grayscale_image, (5,5), 1)
  cv2.imwrite('Q1-Output/blurred.jpeg', blurred)
  edges = canny_edge(blurred, 50, 50, False)
  cv2.imwrite('Q1-Output/canny_edges.jpeg', edges)
  results = fill_accumulator_map(edges, 100, 125, 5)
  valid_circles = get_circles_with_threshold(results, threshold = 0.4, num_points = 360)
  for each_circle in valid_circles:
    y, x, r = each_circle
    cv2.circle(img, (y,x), r, (0,0,255), 1)
  cv2.imwrite('Q1-Output/circles.jpeg', img)