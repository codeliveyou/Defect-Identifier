import numpy as np
import cv2
import os


image_dir = "./dfscer43t/photoset1"
bmp_images = [f"{i:02d}.bmp" for i in range(1, 25)]
large_image = "S10036880#3.tif"

# image_dir = "./dfscer43t/photoset2"
# bmp_images = [f"{i:02d}.bmp" for i in range(1, 25)]
# large_image = "S10036880#6.tif"

# image_dir = "./dfscer43t/photoset3"
# bmp_images = [f"{i:02d}.bmp" for i in range(1, 21)]
# large_image = "test.jpg"

def analyze_and_store_image(image):
    if image is not None:
        height, width = image.shape
        dtype = image.dtype
        print(f"Image: {os.path.basename(image_path)}")
        print(f"  Dimensions: {width}x{height}")
        print(f"  Data Type: {dtype}")
    else:
        print(f"Failed to load image: {os.path.basename(image_path)}")
    

image_list = []
for image_name in bmp_images:
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image_list.append(image)
    else:
        print(f"Failed to load image: {os.path.basename(image_path)}")


large_image_path = os.path.join(image_dir, large_image)
large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)
if large_image is None:
    print(f"Failed to load image: {os.path.basename(image_path)}")
analyze_and_store_image(large_image)

print(f"Total images loaded: {len(image_list)}")

width_overloop, height_overloop = 0, 0
large_image_height, large_image_width = large_image.shape
small_image_height, small_image_width = image_list[0].shape
scaled_small_image_height, scaled_small_image_width = small_image_height - height_overloop, small_image_width - width_overloop
row_number, column_number = 4, 6
# row_number, column_number = 5, 4
new_image_height, new_image_width = row_number * scaled_small_image_height, column_number * scaled_small_image_width

new_image = np.zeros((new_image_height, new_image_width), dtype=np.uint8)

current_image_id = 0
for current_small_image in image_list:
    row_id, column_id = row_number - 1 - current_image_id % row_number, column_number - 1 - (current_image_id // row_number)
    new_image[row_id * scaled_small_image_height : (row_id + 1) * scaled_small_image_height, column_id * scaled_small_image_width : (column_id + 1) * scaled_small_image_width] = current_small_image[height_overloop:, width_overloop:]
    current_image_id += 1
    print(f"proceed image {current_image_id}")

cv2.imwrite(os.path.join(image_dir, "combination_generated_image.jpg"), new_image)

min_large_image, max_large_image = np.min(large_image), np.max(large_image)
min_new_image, max_new_image = np.min(new_image), np.max(new_image)

_lambda = (max_large_image - min_large_image) / (max_new_image - min_new_image)

rescaled_new_image = (min_large_image - min_new_image * _lambda) + new_image * _lambda

rescaled_new_image = rescaled_new_image.astype(np.uint8)

# rescaled_new_image = cv2.GaussianBlur(rescaled_new_image, (15, 15), 0)

cv2.imwrite(os.path.join(image_dir, 'rescaled_new_image.jpg'), rescaled_new_image)

# result = ""
# height, width = rescaled_new_image.shape
# for i in range(height - 1, height - 750, -1):
#     min_value = np.max(rescaled_new_image[:, i : i + 1])
#     result += str(i) + ', ' + str(min_value) + '\n'
# with open('output.txt', "w") as file:
#     file.write(result)


# exit(0)

mark_x, mark_y = 16810, 560
real_mark_x, real_mark_y = 23835, 2222
# real_mark_x, real_mark_y = 23840, 13006
left_paddle = 655

new_scale = (real_mark_x - left_paddle) / mark_x
scaled_large_image_height = int(large_image_height * (new_scale - 0.009))
# print(new_scale)
scaled_large_image_width = int(large_image_width * new_scale)
scaled_large_image = cv2.resize(large_image, (scaled_large_image_width, scaled_large_image_height))
scaled_mark_x, scaled_mark_y = int(mark_x * new_scale), int(mark_y * new_scale)

design_image = np.full((new_image_height, new_image_width), 255, dtype=np.uint8)


for i in range(-1, 2):
    left_top_x,     left_top_y     = real_mark_x - scaled_mark_x,                            real_mark_y - scaled_mark_y +       i * scaled_large_image_height
    right_bottom_x, right_bottom_y = real_mark_x - scaled_mark_x + scaled_large_image_width, real_mark_y - scaled_mark_y + (i + 1) * scaled_large_image_height
    if right_bottom_y <= 0 or left_top_y >= new_image_height:
        continue
    r_left_top_y     = max(left_top_y,     0)
    r_right_bottom_y = min(right_bottom_y, new_image_height)

    design_image[r_left_top_y : r_right_bottom_y, left_top_x : right_bottom_x] = scaled_large_image[r_left_top_y - left_top_y : r_right_bottom_y - left_top_y, :]

cv2.imwrite(os.path.join(image_dir, 'design_image.jpg'), design_image)

current_image_id = 0
for current_small_image in image_list:
    row_id, column_id = row_number - 1 - current_image_id % row_number, column_number - 1 - (current_image_id // row_number)
    cv2.imwrite(os.path.join(image_dir, f'{(current_image_id + 1):02d}_design.bmp'), design_image[row_id * scaled_small_image_height : (row_id + 1) * scaled_small_image_height, column_id * scaled_small_image_width : (column_id + 1) * scaled_small_image_width])
    current_image_id += 1
    print(f"proceed design image {current_image_id}")


