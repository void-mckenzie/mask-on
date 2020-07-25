# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:28:09 2020

@author: mukmc
"""
from bs4 import BeautifulSoup as bs

def label(obj):
    if(obj.find('name').text == 'with_mask'):
        return(0)
    elif(obj.find('name').text== 'without_mask'):
        return(1)
    else:
        return(2)

def getbox(obj):
    l1=[]
    l1.append(int(obj.find('xmin').text))
    l1.append(int(obj.find('ymin').text))
    l1.append(int(obj.find('xmax').text))
    l1.append(int(obj.find('ymax').text))
    return(l1)

def create_list(ldir):
    
    lab_list=[]
    coords=[]
    imglist=[]
    for i in ldir:
        with open('annotations/'+i) as f:
            data= f.read()
            soup = bs(data,'xml')
            objects = soup.find_all('object')
            for j in objects:
                lab_list.append(label(j))
                coords.append(getbox(j))
                imglist.append('images/'+i.split('.')[0]+'.png')
    return (lab_list,coords,imglist)

import os

l = os.listdir("annotations/")

lab_list,coords,imglist = create_list(l)

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img


def create_folders(lab_list,coords,imglist):
    if(not os.path.exists("data")):
        os.mkdir("data")
        os.mkdir("data/0/")
        os.mkdir("data/1/")
        os.mkdir("data/2/")
        for i in range(0,len(imglist)):
            img = load_img(imglist[i])
            arr = img_to_array(img)
            crop = arr[coords[i][1]:coords[i][3],coords[i][0]:coords[i][2]]
            cropimg = array_to_img(crop)
            save_img(path='data/{}/img_{}.png'.format(lab_list[i],i),x=cropimg,scale=False)
    print("fin")

create_folders(lab_list,coords,imglist)

import split_folders

if(not os.path.exists("actual_data")):
    split_folders.ratio('data', output="actual_data", seed=2492, ratio=(.8, .1, .1))