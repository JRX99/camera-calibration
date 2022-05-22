from asyncio.windows_events import NULL
from cmath import pi
from email import header
import detectron2    
from detectron2.utils.logger import setup_logger
setup_logger()
import matplotlib.pyplot as plt
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pathlib import Path
import os
import glob
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
import math
import time
from tabulate import tabulate
from pyzbar.pyzbar import decode
from loguru import logger
from skimage import filters
from skimage.transform import hough_circle,hough_circle_peaks
import pandas as pd


#usage of predictor and locating main points of interest
def find_corners(image:np.ndarray)->np.ndarray:
    predictor = DefaultPredictor(cfg)
    output = predictor(image)
    corners=[]
    pred_boxes=[]
    masks=[]
    v = Visualizer(image[:, :, ::-1],metadata=pig_leg_surgery_metadata,scale=1,instance_mode=ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    prediction_data=output["instances"].to("cpu")
    number_of_found_objects=output['instances']._fields['pred_classes'].to("cpu").shape[0]
    for obj_no in range(number_of_found_objects):
        if(output['instances']._fields['scores'].to("cpu")[obj_no].item()>0.60): #pokud si je síť jistá
            img_copy=np.copy(image)
            mask=output['instances']._fields['pred_masks'].to("cpu")[obj_no,:,:].numpy()#true false pole
            pred_box=output['instances']._fields['pred_boxes'].to("cpu").tensor.numpy()[obj_no]
            corner=img_copy[int(pred_box[1]):int(pred_box[3]),int(pred_box[0]):int(pred_box[2])]
            pred_boxes.append((pred_box,(output['instances']._fields['pred_classes'][obj_no].item(),output['instances']._fields['scores'].to("cpu")[obj_no].item())))
            if(output['instances']._fields['pred_classes'][obj_no].item()!=4):
                corners.append(corner)
            masks.append(mask)  

    return corners , pred_boxes , masks
    
from skimage import filters
from skimage.transform import hough_circle,hough_circle_peaks
#show predictions generated by model
def show_prediction(image:np.ndarray):
    predictor = DefaultPredictor(cfg)
    output = predictor(image)
    corners=[]
    pred_boxes=[]
    v = Visualizer(image[:, :, ::-1],metadata=pig_leg_surgery_metadata,scale=1,instance_mode=ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()

#hough transform for finding holes
def hough_transform_find_holes(corners:np.ndarray)->float:
    for corner in corners:
        corner_color=np.copy(corner)
        edges = filters.unsharp_mask(corner, radius=5, amount=2)
        edges=np.dot(edges[...,:3], [0.1140, 0.5870, 0.2989])
        edges=np.abs(100*filters.laplace(edges)).astype(np.uint8)
        rmin=edges.shape[0]*0.2//2
        rmax=edges.shape[0]*0.4//2
        hough_radii = np.arange(rmin, rmax, 1)
        hough_res = hough_circle(edges, hough_radii)
        radii = []

        accums, center_x, center_y, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1)
        return radii

#measuring holding table
def measure_table(pred_boxes:np.ndarray,image:np.ndarray,masks:np.ndarray)->np.ndarray:

    lower_left=[]
    upper_left=[]
    lower_right=[]
    upper_right=[]
    holes=[]
    for i in range(len(pred_boxes)):
        boxes=[]
        corner_class1=pred_boxes[i][1][0]
        if corner_class1==4:
            holes.append(masks[i])

        boxes.append(pred_boxes[i])

        for j in range(len(pred_boxes)):
            corner_class2=pred_boxes[j][1][0]
            if(corner_class1==corner_class2 and i!=j):
                pred_boxes[j]
        max=0
        for k in range(len(boxes)):
            if boxes[k][1][1]>boxes[max][1][1]:
                max=k


        if corner_class1==0:
            upper_left.append((boxes[max][0][0],boxes[max][0][1]))
        if corner_class1==1:  
            upper_right.append((boxes[max][0][2],boxes[max][0][1]))  
        if corner_class1==2:    
            lower_left.append((boxes[max][0][0],boxes[max][0][3]))  
        if corner_class1==3:
            lower_right.append((boxes[max][0][2],boxes[max][0][3]))  
        print("UL",upper_left)
        print("UR",upper_right)
        print("LL",lower_left)
        print("LR",lower_right)

    print(holes)
    radiuses=[]
    for i in range(len(holes)):
        S=np.count_nonzero(holes[i]==True)
        r=math.sqrt(S/pi)
        radiuses.append(r)
        
    side_a=0
    side_b=0
    side_c=0
    side_d=0
    diagonal_1=0
    diagonal_2=0


    if(len(lower_right)!=0 and len(lower_left)!=0):
        side_a = math.sqrt((lower_right[0][0]-lower_left[0][0])**2+((lower_right[0][1]-lower_left[0][1]))**2)
    if(len(lower_right)!=0 and len(upper_right)!=0):
        side_b = math.sqrt((lower_right[0][0]-upper_right[0][0])**2+((lower_right[0][1]-upper_right[0][1]))**2)
    if(len(upper_right)!=0 and len(upper_left)!=0):
        side_c = math.sqrt((upper_right[0][0]-upper_left[0][0])**2+((upper_right[0][1]-upper_left[0][1]))**2)
    if(len(lower_left)!=0 and len(upper_left)!=0):
        side_d = math.sqrt((lower_left[0][0]-upper_left[0][0])**2+((lower_left[0][1]-upper_left[0][1]))**2)

    print("A,C:",side_a,side_c)
    print("B,D:",side_b,side_d)
    if(len(lower_right)!=0 and len(upper_left)!=0):
        diagonal_1 = math.sqrt((lower_right[0][0]-upper_left[0][0])**2+((lower_right[0][1]-upper_left[0][1]))**2)
    if(len(upper_right)!=0 and len(lower_left)!=0):    
        diagonal_2 = math.sqrt((lower_left[0][0]-upper_right[0][0])**2+((lower_left[0][1]-upper_right[0][1]))**2)
    print("diagonals:",diagonal_1,diagonal_2)
    print()
    metrics=[side_a,side_b,side_c,side_d,diagonal_1,diagonal_2]
    return metrics,radiuses

#method for configuring neural network model
def configure_network():
    
    cfg.merge_from_file(model_zoo.get_config_file("detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "PATH_TO_MODEL")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   
    cfg.DATASETS.TEST = ("pig_leg_surgery", )
    cfg.DATASETS.TEST = ()   
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.03
    cfg.SOLVER.MAX_ITER =1  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

#main calibration method
def calibrate(image:np.ndarray,real_sideAC_mm:int=320,real_sideBD_mm:int=135,real_diag_mm:int=350,real_r_mm:float=5.5)->np.ndarray:
    """
    Method for evaluation of measurements. 

    Args:
    image (ndarray): image loaded in Numpy Array, 
    real_sideAC_mm (int): size of longer side of underlaying pad in millimeters. 
    real_sideBD_mm (int): size of shorter side of underlaying pad in millimeters.
    real_diag_mm (int): size of diagonal of underlaying pad in millimeters.
    real_r_mm (float): radius of opening in underlaying pad in millimeters.
    
    Output is array,
    output_list: contains all measurements

    """
    
    #show_predictions(image)
    start_time=time.perf_counter()
    corners,pred_boxes,masks=find_corners(image)
    end_time=time.perf_counter()
    fincorn_time=end_time-start_time
    start_time=time.perf_counter()
    metrics,radiuses_NN=measure_table(pred_boxes,image,masks)
    end_time=time.perf_counter()
    meastable_time=end_time-start_time
    start_time=time.perf_counter()
    radiuses_HT=hough_transform_find_holes(corners)
    end_time=time.perf_counter()
    hough_time=end_time-start_time
    start_time=time.perf_counter()
    pix_size_QR=main_qr(image)
    end_time=time.perf_counter()
    qr_time=end_time-start_time
    radius_pix_to_mm=0
    sides_pix_to_mm=0
    non_null_metrics=0
    for x in range(len(metrics)):
        if metrics[x]!=0:
            if x==0 or x==1:
                sides_pix_to_mm+=metrics[x]/real_sideAC_mm
                non_null_metrics+=1
            if x==2 or x==3:
                sides_pix_to_mm+=metrics[x]/real_sideBD_mm
                non_null_metrics+=1
            if x==4 or x==5:
                sides_pix_to_mm+=metrics[x]/real_diag_mm
                non_null_metrics+=1
    if non_null_metrics!=0:
        sides_pix_to_mm=sides_pix_to_mm/non_null_metrics           
    else:
        sides_pix_to_mm=0

    radiuses_HT=radiuses_HT[0]/real_r_mm
    for x in range(len(radiuses_NN)):
        radius_pix_to_mm += radiuses_NN[x]/real_r_mm
    if len(radiuses_NN)!=0:
        radius_pix_to_mm=radius_pix_to_mm/len(radiuses_NN)
    else:
        radius_pix_to_mm=0
    if pix_size_QR==None:
        pix_size_QR=0
    output_list=[pix_size_QR,radius_pix_to_mm,sides_pix_to_mm,radiuses_HT,fincorn_time,meastable_time,hough_time,qr_time]
    return output_list
    
#method for reading QR codes - 
def main_qr(img:np.ndarray)->float:
    
    
    QRinit = False
    pix_size = 1.0
    qr_size = 0.027
    is_detected = 0
    img_first = None
    box = []
    qr_text = None
    qr_scissors_frame = []
    i = -1
    #try read QR code
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = decode(grey)
    for oneqr in res:
            txt = oneqr.data.decode("utf8")
            logger.debug(f"qr code text = '{txt}', frame={i}")
            if txt == "Resolution 30 mm":
                qr_size = 0.030
                qr_text = txt
            elif txt == "QR scale pigleg":
                qr_size = 0.027
                qr_text = txt
            elif txt == "Scissors 30 mm":
                qr_scissors_frame.append(i)
                if qr_text is None:
                    # Use only if no Scale QR code was detected
                    qr_size = 0.030
                    qr_text = txt
            else:
                logger.debug(f"Unknown QR code with text='{txt}', on frame={i}")
                continue
            is_detected = 1
            a = np.array(oneqr.polygon[0])
            b = np.array(oneqr.polygon[1])
            #print(a,b)
            qr_side = math.sqrt((a[0]-b[0])**2+((a[1]-b[1]))**2)
            pix_to_mm = qr_side/(qr_size*1000)
            return pix_to_mm
    

# main method
if __name__ == '__main__':
    
    start_time=time.perf_counter()
    register_coco_instances("pig_leg_surgery", {}, Path(f"PATH_TO_JSON_FILE"), Path(f"PATH_TO_IMAGES"))
    pig_leg_surgery_metadata = MetadataCatalog.get("pig_leg_surgery")
    dataset_dicts = DatasetCatalog.get("pig_leg_surgery")
    cfg = get_cfg()
    configure_network()
    calibrate()