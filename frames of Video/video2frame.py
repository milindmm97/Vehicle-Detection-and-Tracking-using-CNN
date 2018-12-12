# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:13:13 2018

@author: Milind
"""

import cv2 

videocap = cv2.VideoCapture('project_video.mp4')

success,image = videocap.read()

count = 0

while success:
    success,image = videocap.read()
    cv2.imwrite("frame%d.jpg" % count, image)
    if cv2.waitKey(10) == 27:
         break
    count += 1