import os
import cv2
import numpy as np
bg = r'C:\Users\sazgar\Downloads\578 172183240801\23-26-44_rgb.png'
ov = r'C:\Users\sazgar\Downloads\578 172183240801\23-26-44_dybde.png'
out = r'out\Gergiev_art_helps_putin_to_kill_gergiev_is_war_supporter.JPG'

# bg = r'C:\Users\sazgar\Downloads\578 172183240801\23-27-28_rgb.png'
# ov = r'C:\Users\sazgar\Downloads\578 172183240801\23-27-28_dybde.png'

path = "/home/skh018/PycharmProjects/MixMatch_new/habib/udderRGB_Depth2/udderRGB_Depth2/val/0/"
# path = "C:\\Users\sazgar\Downloads\For Salman\For Salman\\"
for f in os.listdir(path):
    if f.endswith('rgb.png'):
        bg = path + "/"+f
        # ov = path + f.split('_')[0] + '_dybde.png'
        out = path + "/" + f.split('.')[0] + '_crop.png'



        assert os.path.isfile(bg)
        # assert os.path.isfile(ov)
        background = cv2.imread(bg)
        # b_h, b_w, b_ch = background.shape
        # print(b_h, b_w, b_ch)

        # overlay = cv2.imread(ov)
        # o_h, o_w, o_ch = overlay.shape
        # print(o_h, o_w, o_ch)
        new_b_h = int(480 * 1.1)
        new_b_w = int (640*1.45)
        new_o_h = 480
        new_o_w = 640
        new_background = cv2.resize(background,(new_b_w, new_b_h))
        new_background = new_background[new_b_h - 30 - new_o_h:new_b_h - 30, new_b_w - 140 - new_o_w: new_b_w - 140]
        # cv2.imshow('adjusted', new_background)
        cv2.imwrite(out, new_background)
        # cv2.waitKey()



                    
                    