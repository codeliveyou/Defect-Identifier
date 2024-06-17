import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import json

large_image_height, large_image_width = 11600, 16890
row_number, column_number = 4, 6
real_mark_x, real_mark_y = 23835, 2222
mark_x, mark_y = 16810, 560
left_paddle = 655
new_scale = (real_mark_x - left_paddle) / mark_x
scaled_large_image_height = int(large_image_height * (new_scale - 0.009))
# print(new_scale)
scaled_large_image_width = int(large_image_width * new_scale)
scaled_mark_x, scaled_mark_y = int(mark_x * new_scale), int(mark_y * new_scale)

def get_original(x, y, w, h):
    return {
        "point" : [x * large_image_width // scaled_large_image_width, y * large_image_height // scaled_large_image_height],
        "length" : [w * large_image_width // scaled_large_image_width, h * large_image_height // scaled_large_image_height]
        }


def comparator(img_dir, design_img_name, product_img_name):
    design_img_path = os.path.join(img_dir, design_img_name)
    product_img_path = os.path.join(img_dir, product_img_name)
    result_path = os.path.join(img_dir, f"{product_img_name.split('.')[0]}_result.bmp")

    design_img = cv2.imread(design_img_path)
    product_img = cv2.imread(product_img_path)

    design_gray = cv2.cvtColor(design_img, cv2.COLOR_BGR2GRAY)
    product_gray = cv2.cvtColor(product_img, cv2.COLOR_BGR2GRAY)

    height, width = design_gray.shape
    _, white_mask = cv2.threshold(design_gray, 254, 255, cv2.THRESH_BINARY)
    delta = 15
    
    # cv2.imshow('White Mask', cv2.resize(white_mask, (1024, 1024)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(0)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(design_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(product_gray, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print(f"Not enough matches are found - {len(good_matches)}/4")
        cv2.imwrite(result_path, product_img)
        print(f"Saved result to {result_path}")
        return

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if M is None:
        print(f"Homography could not be computed for {product_img_name}")
        cv2.imwrite(result_path, product_img)
        print(f"Saved result to {result_path}")
        return

    height, width = design_gray.shape
    aligned_product_img = cv2.warpPerspective(product_img, M, (width, height))

    aligned_product_gray = cv2.cvtColor(aligned_product_img, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(design_gray, aligned_product_gray, full=True)
    diff = (diff * 255).astype("uint8")

    _, thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_positions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 80 or h > 80:
            continue
        if np.min(white_mask[max(y - delta, 0) : min(y + h + delta, height), max(x - delta, 0) : min(x + w + delta, width)]) == 0:
            continue

        result_positions.append([x, y, w, h])
        cv2.rectangle(product_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.rectangle(design_image, (x + , y + ), (x + w + width * column_id, y + h + height * row_id), (0, 0, 255), 2)
        defect_positions.append(get_original(x - (real_mark_x - scaled_mark_x) + width * column_id, (y - (real_mark_y - scaled_mark_y) + height * row_id + scaled_large_image_height) % scaled_large_image_height, w, h))
    
    cv2.imwrite(result_path, product_img)
    print(f"Saved result to {result_path}")
    return result_positions


image_dir = "./dfscer43t/photoset1"
defect_positions = []

design_image = cv2.imread(os.path.join(image_dir, "S10036880#3.tif"))
row_id, column_id = 0, 0

for i in range(0, 24):
    row_id, column_id = row_number - 1 - i % row_number, column_number - 1 - (i // row_number)
    comparator(image_dir, f"{(i + 1):02d}_design.bmp", f"{(i + 1):02d}.bmp")
    print(f"Processed image {i+1:02d}")

for rectangle in defect_positions:
    cv2.rectangle(design_image, (rectangle['point'][0], rectangle['point'][1]), (rectangle['point'][0] + rectangle['length'][0], rectangle['point'][1] + rectangle['length'][1]), (0, 0, 255), 2)

cv2.imwrite(os.path.join(image_dir, "result.tif"), design_image)
with open(os.path.join(image_dir, "defect_positions.json"), "w") as json_file:
    json.dump(defect_positions, json_file, indent=4)
