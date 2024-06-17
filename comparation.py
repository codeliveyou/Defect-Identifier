import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the design and product images
design_img = cv2.imread('01_design.bmp', cv2.IMREAD_GRAYSCALE)
product_img = cv2.imread('01.bmp', cv2.IMREAD_GRAYSCALE)

# Ensure the images have been loaded correctly
if design_img is None or product_img is None:
    print("Error: One or both images not found.")
    exit()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(design_img, None)
keypoints2, descriptors2 = sift.detectAndCompute(product_img, None)

# Match descriptors using FLANN matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Ensure that we have enough matches
if len(good_matches) > 10:
    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # Warp design image to align with the product image
    h, w = design_img.shape
    aligned_design = cv2.warpPerspective(design_img, M, (w, h))

    # Compute the absolute difference between aligned design and product image
    difference = cv2.absdiff(aligned_design, product_img)

    # Apply a binary threshold to the difference image to highlight the errors
    _, thresholded = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Find contours of the thresholded differences
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the product image to mark the errors
    error_count = 0
    output_img = cv2.cvtColor(product_img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        # Ignore small contours that may be noise
        if cv2.contourArea(contour) > 50:
            error_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the number of errors
    print(f"Number of errors detected: {error_count}")

    # Display the images
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title('Design Image')
    plt.imshow(design_img, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Product Image')
    plt.imshow(product_img, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('Aligned Design Image')
    plt.imshow(aligned_design, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('Errors Marked')
    plt.imshow(output_img)

    plt.show()

else:
    print("Not enough matches are found - {}/{}".format(len(good_matches), 10))
