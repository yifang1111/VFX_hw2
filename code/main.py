import numpy
import cv2
import os
import glob
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.ndimage import maximum_filter
import argparse

def fixed_random(seed=2322):
    random.seed(seed)
    np.random.seed(seed)


def load_data_and_f(data_name):
    root = os.getcwd()
    img_path = os.path.join(root, 'data', data_name,"*.JPG")
    filenames = sorted(glob.glob(img_path))
    images = []
    for fn in filenames:
        images.append(cv2.imread(fn))

    focal_path = os.path.join(root, 'data', data_name,"focal.txt")
    with open(focal_path, "r") as f:
        focals = f.read().splitlines()
    focals = sorted(focals)
    focals = [float(f.split(" ")[1]) for f in focals] #

    return images, focals, filenames


def cyclindrical(img, f):
    h, w = img.shape[0], img.shape[1]
    proj = np.zeros(img.shape, np.uint8)
    for x in range(-w//2, w-w//2):
        for y in range(-h//2, h-h//2):
            proj_x = round(f*math.atan(x/f)) + w//2
            proj_y = round(f*y/math.sqrt(x**2 + f**2)) + h//2
            proj[proj_y][proj_x] = img[y + h//2][x + w//2]
    return proj


def Harris_detector(img, img_num, save=False, k=0.04, threshold=0.01):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (5,5), 0)
    Iy, Ix = np.gradient(gaussian)
    Sx2 = cv2.GaussianBlur(Ix**2, (5,5), 0)
    Sy2 = cv2.GaussianBlur(Iy**2, (5,5), 0)
    Sxy = cv2.GaussianBlur(Ix*Iy, (5,5), 0)
    R = np.zeros((img.shape[0],img.shape[1]))
    det = Sx2*Sy2-Sxy*Sxy
    tr = Sx2+Sy2
    R = det-k*(tr**2)

    R[R<threshold*np.max(R)] = 0
    #NMS
    R *= (R == maximum_filter(R, footprint=np.ones((3, 3))))
    R = np.where(R>0)

    kp = []
    img_det = img.copy()
    for pt in zip(*R[::-1]):
        kp.append(pt)
        cv2.circle(img_det, pt, 1, (0, 0, 255), -1)
    print('kp_num',len(kp))

    if save:
        cv2.imwrite(f"./report_img/detection_{img_num}.jpg", img_det)

    return kp


def histogram(img, num_bins, sigma):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    L = cv2.GaussianBlur(gray, (5,5), 0)
    Iy, Ix = np.gradient(L)
    m = np.sqrt(Ix**2 + Iy**2)
    theta = np.arctan2(Iy, Ix)*180/np.pi
    theta[theta<0] = theta[theta<0]+360
    
    bin_width = 360//num_bins 
    bucket = np.floor(theta//bin_width)
    # hist = np.zeros((num_bins, img.shape[0], img.shape[1]))
    # m = cv2.GaussianBlur(m, (5,5), sigma)
    # for b in range(num_bins):
    #     hist[b][bucket==b] = m[bucket==b]
   
    return bucket


def SIFT_descriptor(img, kp):
    ori_hist = histogram(img, 36, 1.5)
    # ori = np.argmax(ori_hist, axis=0)*10+5
    ori = ori_hist*10+5

    feat = []
    h, w = img.shape[0], img.shape[1]
    for x, y in kp:
        x=int(x)
        y=int(y)
        M = cv2.getRotationMatrix2D((x, y), ori[y][x], 1.0)
        rimg = cv2.warpAffine(img, M, (w, h))
        hist = histogram(rimg, 8, 0.5*16)
        desc = []
        if y-8>=0 and y+8<h and x-8>=0 and x+8<w:
            for i in range(x-8,x+8,4):
                for j in range(y-8,y+8,4):
                    cnt = []
                    for b in range(8):
                        cnt.append(np.sum(hist[j:j+4, i:i+4]==b))
                    desc.extend(cnt)

            if np.linalg.norm(desc)!=0:
                desc = desc/np.linalg.norm(desc)          
                desc[desc>0.2] = 0.2
                desc = desc/np.linalg.norm(desc)         
            feat.append(((x, y), desc))
            
    return feat


def feature_matching(feat1, feat2, threshold = 0.3):
    matches = []
    for i in range(len(feat1)):
        d = []
        for j in range(len(feat2)):
            diff = np.linalg.norm(feat1[i][1] - feat2[j][1], ord=2)
            d.append((j,diff))
        d = sorted(d, key=lambda x:x[1])
        best_kp, min_diff = d[0]
        _, sec_diff = d[1]
        # print(min_diff)
        if min_diff < threshold and min_diff < 0.8 * sec_diff:
            matches.append((feat1[i][0], feat2[best_kp][0]))
    print('match_num',len(matches))
    return matches  


def RANSAC_best_moving(match_result, threshold = 5):# voting_threshold: square_distance
    # get the shift of x and y
    shifts = []
    for L, R in match_result:
        shifts.append( (L[0] - R[0], L[1] - R[1]) )

    # get the valid shifts 
    voting_result = []
    for i in range(len(shifts)):
        cadidate = shifts[i]
        all_voting_pair = list(map(lambda x: (x[0] - cadidate[0], x[1] - cadidate[1]), shifts))
        distance_square = list(map(lambda x: x[0] * x[0] + x[1] * x[1], all_voting_pair))
        vote_yes = (np.count_nonzero(np.array(distance_square) < threshold))
        voting_result.append(vote_yes)
    # get the shift with max distance
    best_moving_index = (np.argmax(np.array(voting_result)))
    best_moving = shifts[best_moving_index]
    best_match = match_result[best_moving_index]
    # print(best_match)
    return best_moving, best_match


def padding_img(img, move_x, move_y):
    if(move_x >= 0 and move_y >= 0):
        new_img = np.pad(img, ((move_y, 0), (move_x, 0), (0, 0)), 'constant')
    elif(move_x >= 0 and move_y < 0):
        new_img = np.pad(img, ((0, -move_y), (move_x, 0), (0, 0)), 'constant')
    elif(move_x < 0 and move_y >= 0):
        new_img = np.pad(img, ((move_y, 0), (0, -move_x), (0, 0)), 'constant')
    else:
        new_img = np.pad(img, ((0, -move_y), (0, -move_x), (0, 0)), 'constant')
    return new_img


def shift_warping_and_stitching(best_moving, img_L, img_R, img_num, save=False):
    # img_R: source
    # img_L: dest
    move_x, move_y = best_moving
    # we assert move_x >= 0
    if move_x < 0:
        img_L, img_R = img_R, img_L
        move_x = -move_x
        move_y = -move_y
        print("swap")
    #y, w
    hl,wl,_ = img_L.shape
    hr,wr,_ = img_R.shape

    # <0: 左上位置padding, >0:右下位置padding
    L_padding_x = (wr - wl + move_x)
    L_padding_y = move_y 
    new_img_L = padding_img(img_L, -L_padding_x, -L_padding_y)

    R_padding_x = move_x
    if move_y <= 0:
        R_padding_y = move_y - (hl - hr)
        # print("A", R_padding_y)
        new_img_R = padding_img(img_R, R_padding_x, R_padding_y)

    elif 0 < move_y and move_y <= hl - hr:
        upper_pad = move_y
        lower_pad = hl-hr-move_y+1
        # print("B", upper_pad, lower_pad)
        if move_x >= 0 :
            new_img_R = np.pad(img_R, ((upper_pad, lower_pad), (move_x, 0), (0, 0)), 'constant')
        elif move_x >= 0 :
            new_img_R = np.pad(img_R, ((upper_pad, lower_pad), (0, -move_x), (0, 0)), 'constant')

    elif move_y > hl - hr:
        R_padding_y = hl - hr + move_y 
        # print("C", R_padding_y)
        new_img_R = padding_img(img_R, R_padding_x, R_padding_y)
    
    if save:
        cv2.imwrite(f"./report_img/new_img_L_{img_num}.jpg", new_img_L)
        cv2.imwrite(f"./report_img/new_img_R_{img_num}.jpg", new_img_R)

    stitch_img = np.zeros(new_img_L.shape, dtype="uint8")

    # # Warp source (right) image to destinate (left) image by Homography matrix.
    # Blending the result with source (lft) image  
    intersect_range = wl - move_x   
    intersect_cnt = 0  
    for i in trange(0, stitch_img.shape[1]):
        p_left = 1
        p_right = 0
        if np.count_nonzero(new_img_L[:,i] != 0) and np.count_nonzero(new_img_R[:,i] != 0):
            p_left = 1 - (intersect_cnt / intersect_range)
            p_right = (intersect_cnt / intersect_range)
            intersect_cnt += 1
        elif np.count_nonzero(new_img_L[:,i] != 0):
            p_left = 1
            p_right = 0
        elif np.count_nonzero(new_img_R[:,i] != 0):
            p_left = 0
            p_right = 1

        # linear blending with const.
        stitch_img[:,i][:] = p_left * new_img_L[:,i][:]  + p_right* new_img_R[:,i][:]  
        
    return stitch_img


def remove_black_border(img):
    h,w,_=img.shape
    reduced_h, reduced_w = h, w
    # right to left
    for col in range(w-1, -1, -1):
        all_black = True
        for i in range(h):
            if np.count_nonzero(img[i, col]) > 0:
                all_black = False
                break
        if (all_black == True):
            reduced_w = reduced_w - 1
            
    # bottom to top 
    for row in range(h - 1, -1, -1):
        all_black = True
        for i in range(reduced_w):
            if np.count_nonzero(img[row, i]) > 0:
                all_black = False
                break
        if (all_black == True):
            reduced_h = reduced_h - 1
    
    return img[:reduced_h, :reduced_w] 


def draw_matches(imgL, imgR, matches, img_num, save=False):
    src_pts = [m[0] for m in matches]
    dst_pts = [m[1] for m in matches]

    hl,wl,_ = imgL.shape
    hr,wr,_ = imgR.shape

    img_match = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
    img_match[0:hl, 0:wl] = imgL
    img_match[0:hr, wl:] = imgR
    # Draw the match
    for i, _ in enumerate(src_pts):
        pos_l = src_pts[i]
        pos_r = dst_pts[i][0]+wl, dst_pts[i][1]
        cv2.circle(img_match, pos_l, 1, (0, 0, 255), -1) # src 
        cv2.circle(img_match, pos_r, 1, (0, 255, 0), -1) # dest
        cv2.line(img_match, pos_l, pos_r, (255, 0, 0), 1)

    if save:
        cv2.imwrite(f"./report_img/matching_{img_num}.jpg", img_match)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Image Stitching')
    parser.add_argument('--data', default = 'data1')
    args = parser.parse_args()

    fixed_random(2322)
    # os.makedirs("report_img", exist_ok=True) 

    images, focals, image_paths = load_data_and_f(args.data)
    final_img = cyclindrical(images[0], focals[0]) 
    
    # stitch image R(src) to image L(dst)
    for i in trange(1, len(images)):
        imgL = final_img
        imgR = cyclindrical(images[i], focals[i]) 
        
        print(f'Feature detection and matching {i}')
        kpＬ = Harris_detector(imgＬ, i, False)
        featＬ = SIFT_descriptor(imgＬ, kpＬ)
        kpR = Harris_detector(imgR, i, False)
        featR = SIFT_descriptor(imgR, kpR)
        matches = feature_matching(featL, featR)

        draw_matches(imgL, imgR, matches, i, False)

        print(f"Image matching {i}")
        best_moving, best_match = RANSAC_best_moving(matches)

        print(f"Image stitching {i}")
        final_img = shift_warping_and_stitching(best_moving, final_img, imgR, i, False)

        print(f"Cut the black row {i}")
        final_img = remove_black_border(final_img)

        print(f"Saving result {i}. Current image size: ", final_img.shape[:2])
        # cv2.imwrite(f"./report_img/stitch_image_{i}.jpg", final_img)

    print("Done.")
    cv2.imwrite("result.png", final_img)
