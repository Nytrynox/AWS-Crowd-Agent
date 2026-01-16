import cv2
import numpy as np

def motion_spike(prev_gray, curr_gray, threshold=1.5):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mean_mag = float(np.mean(mag))
    return mean_mag > threshold, mean_mag
