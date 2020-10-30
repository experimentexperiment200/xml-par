import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def parse_boundbox(xmlpath):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    boxes_list=[]
    for i,element in enumerate(root.iter('object')):

        box = [0.0]*4
        class_name = element.find('name').text
        bndbx = element.find('bndbox')
        box[0] = int(bndbx.find('xmin').text) - 1
        box[1] = int(bndbx.find('ymin').text) - 1
        box[2] = int(bndbx.find('xmax').text) - 1
        box[3] = int(bndbx.find('ymax').text) - 1
        boxes_list.append((class_name,box))
    return boxes_list

def plot_anomaly(input_image,input_annotation_xml_path):
    ori_img = input_image.copy()
    tagname="undefined"
    annotations_list = parse_boundbox(input_annotation_xml_path)
    w_avg=0
    h_avg=0
    count=0
    for j,lst in enumerate(annotations_list):
        label_text = lst[0]
        count=count+1
        bb = lst[-1]
        x1 = int(bb[0])
        y1 = int(bb[1])
        w = int(bb[2] - x1 + 1)
        h = int(bb[3] - y1 + 1)
        cv2.rectangle(ori_img,(x1,y1),(x1+w,y1+h),(0,255,0),1)
        w_avg=w_avg+(bb[2]-x1+1)
        h_avg=h_avg+(bb[3]-y1+1)
    # plt.imshow(ori_img)
    # plt.show()
        # font=cv2.FONT_HERSHEY_SIMPLEX
        # tagname=label_text
        # yy1=y1-20
        # ori_img=cv2.resize(ori_img,(512,512))
        # cv2.putText(ori_img,label_text,(x1,yy1),font,1,(0,255,0),1,cv2.LINE_AA)
    w_avg = w_avg/count
    h_avg = h_avg / count
    print(h_avg/w_avg,h_avg/w_avg)
    # print(w_avg / h_avg)
    return ori_img,tagname,x1,y1,w,h,w_avg,h_avg


def resize_annotations(input_image,input_annotation_xml_path,basename):
    ori_img = input_image.copy()
    img = cv2.resize(ori_img, (300, 300))
    tagname="undefined"
    annotations_list = parse_boundbox(input_annotation_xml_path)
    print("len of annotations_list=",len(annotations_list))





    for j,lst in enumerate(annotations_list):
        label_text = lst[0]
        bb = lst[-1]

        (rows, cols, _) = ori_img.shape
        sx = 300 / rows
        sy = 300 / cols

        scale = (sy, sx)
        x1 = round(bb[0]*scale[0])
        y1 = round(bb[1]*scale[1])
        x2 = round(bb[2]*scale[0])
        y2 = round(bb[3]*scale[1])


        x1 = int(x1)
        y1 = int(y1)
        w = int(x2 - x1 + 1)
        h = int(y2 - y1 + 1)
        mydict = ({'Name': basename, 'x': str(x1), 'y': str(y1), 'w': str(w), 'h': str(h)})
        df = pd.DataFrame(mydict, index=[1])
        with open('resized_annotations.csv', 'a', newline='') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
        # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 1)
    # plt.imshow(img)
    # plt.show()



imagedir='F:/interview/labelmg/Train/resize_images/'
opdir='F:/interview/code/rsna-pneumonia-master/rsna-pneumonia-master/visualise/'
anno_dir='F:/interview/labelmg/Train/resize_annotations/'

def fol_contents(direct):
    return [os.path.join(direct,pth) for pth in os.listdir(direct) ]
ww=0
hh=0
for ipth in fol_contents(imagedir):

    basename = os.path.basename(ipth)
    basename = basename.rsplit('.JPG')[0]

    if basename == "Thumbs":
        continue

    xmlpath = os.path.join(anno_dir,basename+'.xml')
    # print(xmlpath)
    if(os.path.exists(xmlpath)):

        final_img,tag_name,x,y,w,h,w_avg,h_avg = plot_anomaly(cv2.imread(ipth),xmlpath)
        # print(w_avg,h_avg)
        ww=ww+w_avg
        hh=hh+h_avg
        # cv2.imwrite(os.path.join(opdir,basename+'.JPG'),final_img)
# print(ww)
# print(hh)
# print(ww/len(fol_contents(imagedir)))
# print(hh/len(fol_contents(imagedir)))

# def resize_plot():
#     imagedir = 'F:/interview/labelmg/Train/JPEGImages/'
#     anno_dir = 'F:/interview/labelmg/Train/Annotations/'
#     count=0
#     def fol_contents(direct):
#         return [os.path.join(direct, pth) for pth in os.listdir(direct)]
#
#
#     import csv
#
#     # with open('resized_annotations.csv','a') as csvfile:
#     #     fields=['Name','x','y','w','h']
#     #     writer=csv.DictWriter(csvfile,fieldnames=fields)
#     #     writer.writeheader()
#
#     for ipth in fol_contents(imagedir):
#
#         basename = os.path.basename(ipth)
#         basename = basename.rsplit('.JPG')[0]
#
#         if basename == "Thumbs":
#             continue
#
#         xmlpath = os.path.join(anno_dir, basename + '.xml')
#         # print(xmlpath)
#         if (os.path.exists(xmlpath)):
#             count=count+1
#             resize_annotations(cv2.imread(ipth), xmlpath,basename)
#     # print(count)
#
#
# # resize_plot()
#
#