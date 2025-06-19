#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
saurabh kashid (saurabh.kashid@wpi.edu)
Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:
import os 
import numpy as np
import cv2
import argparse
import random
import math
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# Add any python libraries here

def read_images(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)
        images.append(img)
    return images

def harris_corner(gray_images):
    for gray in gray_images:
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        # img[dst>0.01*dst.max()]=[0,0,255]
        
def find_regional_maxima(corner_sc_img, min_distance=3):
    """Finds regional maxima using peak_local_max."""
    maxima = peak_local_max(corner_sc_img, min_distance=min_distance)
    mask = np.zeros_like(corner_sc_img, dtype=bool)
    mask[tuple(maxima.T)] = True
    return mask, maxima

# def apply_anms(corner_score_img, local_max_idx, n_best_corners):
#     distances = np.full(local_max_idx.shape[0], np.inf)
#     for idx, i in enumerate(local_max_idx):
#         for j in local_max_idx:
#             if corner_score_img[i[0],i[1]]<corner_score_img[j[0],j[1]]:
#                 ED = (j[0]-i[0])**2 + (j[1]-i[1])**2
#                 if distances[idx]>ED:
#                     distances[idx] = ED
#     # pick n best
#     top_n_indices = np.argsort(distances)[::-1][:n_best_corners]
#     selected_corners = local_max_idx[top_n_indices]

#     return selected_corners

def apply_anms(corner_score_img, local_max_idx, n_best_corners):
    # Precompute corner scores
    scores = corner_score_img[local_max_idx[:, 0], local_max_idx[:, 1]]
    
    # Initialize suppression radii (distances)
    distances = np.full(local_max_idx.shape[0], np.inf)
    
    # For each corner, find distance to nearest stronger corner
    for idx_i, (coord_i, score_i) in enumerate(zip(local_max_idx, scores)):
        # Find indices of stronger corners
        stronger_idx = np.where(scores > score_i)[0]
        
        if stronger_idx.size > 0:
            coord_j = local_max_idx[stronger_idx]
            dx = coord_j[:, 0] - coord_i[0]
            dy = coord_j[:, 1] - coord_i[1]
            squared_distances = dx**2 + dy**2
            
            # Keep the minimum distance to stronger corner
            distances[idx_i] = np.min(squared_distances)
        else:
            # If no stronger corner, keep distance as np.inf
            distances[idx_i] = np.inf

    # Pick n_best_corners with largest suppression radius
    top_n_indices = np.argsort(distances)[::-1][:n_best_corners]
    selected_corners = local_max_idx[top_n_indices]

    return selected_corners

def draw_corners(img, corner_cords, file_path, display= True):
    img_cp = np.copy(img)
    for y,x in corner_cords:
        cv2.drawMarker(img_cp,(x,y), [0, 255, 0])
    # save the image to the given path 
    cv2.imwrite(file_path,img_cp)
    if display:
        plt.imshow(cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB))

def get_patch(img, coord, patch_size, mode='valid'):
    # padd the img by the patch
    h, w = img.shape[:2]
    x, y = coord
    img_cp = np.copy(img)
    half = patch_size//2
    if mode == 'valid':
            if y - half < 0 or y + half >= h or x - half < 0 or x + half >= w:
                return None  # skip this corner – it doesn't fit
            patch = img[y-half:y+half+1, x-half:x+half+1].copy()
    if mode == 'pad':
        pad_img = cv2.copyMakeBorder(img_cp, half, half, half, half,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=0)
        x, y = coord
        # shift by +half because of the border we added
        y_pad, x_pad = y + half, x + half
        patch = pad_img[y_pad-half:y_pad+half+1, x_pad-half:x_pad+half+1].copy()

    return patch

def standardize_vector(vect):
    mean = np.mean(vect)
    std = np.std(vect)
    std_vect = np.array([(i-mean)/std for i in vect])
    return (vect-mean)/std
    
def feature_descriptors(img, corner_coords, gaussin_blur_patch_size, subsampel_path_size, debug=False):
    features = []
    valid_corners = []
    for idx, (cord_y, cord_x) in enumerate(corner_coords):
        # get the patch 
        patch = get_patch(img, (cord_x,cord_y), gaussin_blur_patch_size)
        if patch is None:
            # skip this corner feataure 
            print("Feature is near to edge of the image skipping")
            continue
        # add the gaussian blur to this patch
        blur = cv2.GaussianBlur(patch, (3,3),0)
        # plot the result 
        # plt.imshow(blur,cv2.COLOR_BGR2RGB)
        print("check gaussian blur")
        if debug:
            fig, axs = plt.subplots(1, 2, figsize=(16, 4))
            axs[0].imshow(patch,cmap="gray")
            axs[0].set_title('patch')

            axs[1].imshow(blur,cmap="gray")
            axs[1].set_title("blur")

        # reize the blur to 8*8 patach
        sub_sample_blur = cv2.resize(blur,(subsampel_path_size,subsampel_path_size), interpolation=cv2.INTER_AREA)
        feature_vect = sub_sample_blur.flatten()
        # standardize the vector to have the 0 mean and 1 variance
        std_feat_vec = standardize_vector(feature_vect)
        features.append(std_feat_vec)
        valid_corners.append([cord_x, cord_y])
    return np.array(features), np.asanyarray(valid_corners)

def ransac_homography(kp1, kp2, feature_map, max_iters=1000):
    kps_src = np.array([point.pt for point in kp1])[list(feature_map.keys())]
    kps_dst = np.array([point.pt for point in kp2])[list(feature_map.values())]
    
    # add 1 in the end to make it 3,1
    one_mat = np.ones((kps_src.shape[0],1))
    kpt_src_homo = np.hstack((kps_src,one_mat)).T # shape 3*n
    max_inliers = []
    for iter in range(max_iters):
        if len(max_inliers)>len(kps_dst):
            break

        # pick random 4 points 
        rand_pts_idx = random.sample(range(len(kps_src)-1),4)
        selected_kpt_src = kps_src[rand_pts_idx]
        selected_kpt_dst = kps_dst[rand_pts_idx]

        H = cv2.findHomography(selected_kpt_src,selected_kpt_dst)
        if H[0] is None:
            continue
        x_transformed = H[0]@kpt_src_homo
        try:
            x_transformed = np.array([x_transformed[0]/x_transformed[2],x_transformed[1]/x_transformed[2]]).T # normalize the x,y with z
        except RuntimeError:
            continue
        error = np.sum((kps_dst - x_transformed)**2, axis=1)

        curr_inliers_no = np.sum(error < 5)
        if curr_inliers_no > np.sum(max_inliers):
            max_inliers = (error < 5)
    
    return max_inliers

def feature_matching(features1, features2, threshold):
    # all the features are 64,1 vect
    feature_map = {}
    dist = np.full((len(features1)),np.inf)
    for idx, f1 in enumerate(features1):

        # for f2 in features2:
            # dist[idx] = math.dist(f1,f2)
        dist = np.sum((features2 - f1)**2, axis=1)
        sort_idx = np.argsort(dist)
        best_match, sec_best_match = dist[sort_idx[0]], dist[sort_idx[1]]
        if (best_match/sec_best_match)<threshold:
            # keep that feature correspondance
            feature_map[idx] = sort_idx[0]
        
    return feature_map

def stitch_two_images(img1, img2, H):
    # img1 → source image (to be warped), img2 → destination (reference) image

    # 1. Compute the size of the final canvas:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # corners of img1
    corners_img1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    # warp them
    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

    # corners of img2
    corners_img2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)

    # all corners together
    all_corners = np.vstack((warped_corners_img1, corners_img2))

    # find bounding box
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # translation to shift by
    translation = [-xmin, -ymin]
    # build translation homography
    H_trans = np.array([[1, 0, translation[0]],
                        [0, 1, translation[1]],
                        [0, 0, 1]])

    # 2. Warp img1 into the panorama canvas
    pano_size = (xmax - xmin, ymax - ymin)
    panorama = cv2.warpPerspective(img1, H_trans.dot(H), pano_size)

    # 3. Paste img2 into the panorama canvas
    panorama[translation[1]:h2 + translation[1],
             translation[0]:w2 + translation[0]] = img2

    return panorama

def corners_after_warp(img, H):
    h, w = img.shape[:2]
    # 4 homogeneous corners
    pts = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]]).T
    warped = H.dot(pts)
    warped /= warped[2]           # normalize
    return warped[:2].T           # Nx2 array

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=200, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--minDistForLocalMax', default=4, help='Minimum distance arg for peak_local_max')
    Parser.add_argument('--outputFilePath', default='YourDirectoryID_p1/Phase1/output_imgs', help='output path to save the imgs')
    Parser.add_argument('--patchSizeForGaussianBlur', default=41, help='patch size for the gaussian blur arounf the feature')
    Parser.add_argument('--subsamplePatchSize', default=8, help='sub-sample the patch size to reduce the complexity')
    Parser.add_argument('--correspondenceThreshold', default=0.8, help='helps to pick the corrospondences')

    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    folder_path = r"YourDirectoryID_p1/Phase1/Data/Train/Set1"
    images = read_images(folder_path)
    # convert to gray scale
    gray_imgs = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32) for img in images]
    img_info = {}
    
    """
	Corner Detection
	Save Corner detection output as corners.png
	"""
    for idx, gray in enumerate(gray_imgs):
        dst_ = cv2.cornerHarris(gray,2,5,0.03)
        dst_ = cv2.dilate(dst_, None)
        
        img_cp = np.copy(images[idx])
        img_cp[dst_>0.01*dst_.max()] = [0,0,255]
        # Display result using matplotlib
        # plt.figure(figsize=(6, 6))
        # plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        # plt.title(f'Harris Corners - Image {idx+1}')
        # plt.axis('off')
        # plt.show(block= True)

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
        corner_response = np.copy(dst_)
        corner_response[corner_response < 0] = 0
        mask, local_maxima_idx = find_regional_maxima(corner_response, min_distance=Args.minDistForLocalMax)
        img_local_max = np.copy(images[idx])
        for (y, x) in local_maxima_idx:
            cv2.drawMarker(img_local_max, (x, y), [0, 255, 0], markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        # plt.figure(figsize=(6,6))
        # plt.imshow(cv2.cvtColor(img_local_max, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.title('Local Maxima after Harris')
        # plt.show()
        n_best_corners = apply_anms(dst_, local_maxima_idx, n_best_corners=NumFeatures)

        # mark the corners and plot
        file_name = f"{Args.outputFilePath}/anm_result_{idx}.png"
        draw_corners(images[idx],n_best_corners,file_name)

        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """
        features, valid_corners = feature_descriptors(gray, n_best_corners, Args.patchSizeForGaussianBlur, Args.subsamplePatchSize)
        img_info[idx] = {"features":features, "valid_corners":valid_corners} # change the features to feature_discripter
    # resize the feature and save 
    print("Done")
    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""
    homography_mat = []
    for i in range(len(images)-1):
        feature_map = feature_matching(img_info[i]["features"],img_info[i+1]["features"],Args.correspondenceThreshold)
        # convert matches to cv2Dmatch
        cv_matches = []
        for query_idx, train_idx in feature_map.items():
            # distance is optional; 0 is fine for visualisation
            cv_matches.append(cv2.DMatch(_queryIdx=query_idx,
                                        _trainIdx=train_idx,
                                        _imgIdx=0,        # not used here
                                        _distance=0.0))
        kp1 = [cv2.KeyPoint(x=float(x), y=float(y), size=1)
        for (x, y) in img_info[i]["valid_corners"]] 

        kp2 = [cv2.KeyPoint(x=float(x), y=float(y), size=1)
            for (x, y) in img_info[i+1]["valid_corners"]]
        vis = cv2.drawMatches(images[i], kp1,
                        images[i+1], kp2,
                        cv_matches, None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        file_name = f"{Args.outputFilePath}/feature_matching_pair_{i+1}.png"
        cv2.imwrite('mypano2.png', vis)
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'{len(cv_matches)} matches (ratio test ≤ 0.64)')
        plt.show()
        """
        Refine: RANSAC, Estimate Homography
        """
        inlier = ransac_homography(kp1,kp2,feature_map)
        # selected inliers 
        valid_kp1 = [kp1[i] for i in feature_map.keys()]
        valid_kp2 = [kp2[i] for i in feature_map.values()]

        kp1_inliers = [valid_kp1[i] for i, flag in enumerate(inlier) if flag]
        kp2_inliers = [valid_kp2[i] for i, flag in enumerate(inlier) if flag]


        # plot the selected keypoints 
        cv_matches = []
        for num in range(len(kp1_inliers)):
            cv_matches.append(cv2.DMatch(_queryIdx=num, _trainIdx=num, _imgIdx=0, _distance=0.0))

        vis = cv2.drawMatches(images[i], kp1_inliers,
                        images[i+1], kp2_inliers,
                        cv_matches, None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'{len(cv_matches)} matches (Ransac)')
        plt.show()

        kps_src = np.array([point.pt for point in kp1_inliers])
        kps_dst = np.array([point.pt for point in kp2_inliers])
        H = cv2.findHomography(kps_src, kps_dst)[0]
        homography_mat.append(H)

        print("test")
    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """
    # 1) First, build cumulative homographies that map img[i] → img[last]
    N = len(images)
    cumH = [None] * N
    cumH[-1] = np.eye(3)                            # last image → last image
    for i in range(N-2, -1, -1):                    # go backwards
        # to map img[i]→last = (img[i+1]→last) ∘ (img[i]→img[i+1])
        cumH[i] = cumH[i+1] @ homography_mat[i]

    # 2) Gather all warped corners to determine panorama extents
    all_corners = []
    for i, img in enumerate(images):
        H = cumH[i]
        all_corners.append(corners_after_warp(img, H))
    all_corners = np.vstack(all_corners)
    x_min, y_min = np.int32(all_corners.min(axis=0) - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0) + 0.5)

    # 3) Compute a translation so everything is in positive coords
    trans = np.array([[1, 0, -x_min],
                    [0, 1, -y_min],
                    [0, 0,     1  ]]) 

    # 4) Create accumulator & weight maps
    W = x_max - x_min
    H = y_max - y_min

    ref_idx = 0  

    # 5) “stamp” + Poisson‐blend
    panorama = np.zeros((H, W, 3), dtype=np.uint8)
    coverage = np.zeros((H,W),dtype=np.uint8)

    for i, img in enumerate(images):
        Htot   = trans @ cumH[i]
        warped = cv2.warpPerspective(img, Htot, (W, H))
        
        # build a mask of *all* non-zero pixels
        # gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        h, w = images[i].shape[:2]
        mask = cv2.warpPerspective(np.ones((h,w),np.uint8)*255, trans@cumH[i], (W,H))
        
        # new_mask     = cv2.bitwise_and(mask, cv2.bitwise_not(coverage))
        # overlap_mask = cv2.bitwise_and(mask, coverage)
        # stamp all valid pixels in
        cv2.copyTo(warped, mask, panorama)

        # panorama[mask>0] = warped[mask>0]
        # coverage |= new_mask
        
        # 5b) Now crop to the bounding box and Poisson‐blend *that* region
        if i == ref_idx:
            continue
        x, y, w_box, h_box = cv2.boundingRect(mask)
        if w_box == 0 or h_box == 0:
            continue

        src = warped[y:y+h_box, x:x+w_box]
        dst = panorama[y:y+h_box, x:x+w_box]
        m   = mask[y:y+h_box, x:x+w_box]

        center = (w_box//2, h_box//2)
        blended = cv2.seamlessClone(src, dst, m, center, cv2.MIXED_CLONE)

        panorama[y:y+h_box, x:x+w_box] = blended
        
        #TODO: resize the panoroma for multiple image stitching
    # 6) Save and display
    cv2.imwrite('mypano1.png', panorama)
    plt.figure(figsize=(20,10))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Stitched Panorama')
    plt.show()

if __name__ == "__main__":
    main()
