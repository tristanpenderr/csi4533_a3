from operator import gt
import cv2 as cv2
import os
import logging
from random import randint
from nn_detector import NN_detector as detector


# log configuration
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

#variables
files = sorted(os.listdir('./img1'))
dict = {}
dir = 'gt.txt'
dir_tracking = 'gt_tracking.txt'
rectangle_englobantes = []
rectangle_englobantes_tracking = []
rectangle_englobantes_tracking_voiture = []
color_dict = {}
img_bounding_boxes = {}

#Object for a bounding
class Box(object):
    x1 = 0.
    x2 = 0.
    y1 = 0.
    y2 = 0.

   # constructor for our box 
    def __init__(self, x, y, l, h, box_type):
        #left
        self.x1 = x
        #right
        self.x2 = x + l
        #bottom
        self.y1 = y
        #top
        self.y2 = y + h

        self.area = h * l

        self.l1 = l
        self.h1 = h
        self.box_type = box_type

# function to create a box
def make_box(x, y, l,h, box_type):
    box = Box(x, y, l , h, box_type)
    return box
#create temporary matrix to be used in the loop 
tmp_matrix = []
tmp_img1 = []
tmp_img2 = []

def populate_variables(files):
    #populate dict with filenames for each picture
    for i in range(len(files)):
        dict[i+1] = 'img1/'+files[i]

    # populate rectange_englobantes_tracking with all the gt_tracking.txt informations
    with open(dir_tracking) as d : 
        for line in d : 
            rectangle_englobantes_tracking.append(line.strip("\n").split(','))

    #populate dict with list of bounding boxes for each picture 
    for i in range(85, 470):
        # définir le chemin d'une image 
        path_to_image = files[i]
        
        print(path_to_image)
        # fixer un seuil pour le score
        score_threshold = 0.3

        # instancier la classe avec la valeur souhaitée pour "GPU_detect"
        det = detector(GPU_detect = False)
         # exécuter l'inférence sur l'image
        detections = det.detect("img1/" + path_to_image)
        # charger l'image dans OpenCV pour la manipuler
        img = cv2.imread("img1/" + path_to_image)
        # boucle sur les détections et filtre les scores faibles
        for i in range(0, len(detections["boxes"])):
            score = detections["scores"][i]
            id = int(detections["labels"][i])
            # ici, 3 correspond à une voiture, 1 à une personne et 6 à un autobus
            if score > score_threshold and (id == 3 or id == 6 or id == 1):
                box = detections["boxes"][i]       
                (x1, y1, x2, y2) = box.astype("int")
                # dessiner les boîtes englobantes sur l'image

                if path_to_image not in img_bounding_boxes :
                    img_bounding_boxes[path_to_image] = [make_box(x1,y1,x2 - x1,y2 - y1, id)]
                else : 
                    img_bounding_boxes[path_to_image] += [make_box(x1,y1,x2 - x1,y2 - y1, id)]
                


        # for j in range(len(rectangle_englobantes)) : 
        #     if rectangle_englobantes[j][0] == str(i + 1) : 
        #         x1 = int(rectangle_englobantes[j][2])
        #         y1 = int(rectangle_englobantes[j][3])
        #         l1 = int(rectangle_englobantes[j][4])
        #         h1 = int(rectangle_englobantes[j][5])
        #         logging.warning((rectangle_englobantes[j][6],rectangle_englobantes[j][7]))
        #         if box_type == "3":
        #             if h1 <= (1.25 * l1) and files[i] not in img_bounding_boxes :  
        #                 img_bounding_boxes[files[i]] = [make_box(x1,y1,l1,h1)]
        #             elif h1 <= (1.25 * l1): 
        #                 img_bounding_boxes[files[i]] += [make_box(x1,y1,l1,h1)]
        #             elif files[i] not in img_bounding_boxes :
        #                 img_bounding_boxes[files[i]] = []
        #         elif box_type == "1":
        #             if (2 * l1) <= h1 and files[i] not in img_bounding_boxes :  
        #                 img_bounding_boxes[files[i]] = [make_box(x1,y1,l1,h1)]
        #             elif (2 * l1) <= h1: 
        #                 img_bounding_boxes[files[i]] += [make_box(x1,y1,l1,h1)]
        #             elif files[i] not in img_bounding_boxes :
        #                 img_bounding_boxes[files[i]] = []
                    
# function for getting image from dict
def get_image(img_num)  :
    return cv2.imread(dict[img_num])

#get all the cars in the gt_tracking file
def get_boxes() : 
    for i in range(len(rectangle_englobantes_tracking)):
        if rectangle_englobantes_tracking[i][7] == 1 or rectangle_englobantes_tracking[i][7] == 3  :
            rectangle_englobantes_tracking_voiture.append(rectangle_englobantes_tracking[i])

# generate random color 
def generate_color() : 
    return [randint(0, 255), randint(0, 255), randint(0, 255)]

def calculate_iou(box1, box2):
    x1, y1 = max(box1.x1,box2.x1), max(box1.y1,box2.y1)
    x2, y2 = min(box1.x2, box2.x2), min(box1.y2, box2.y2)
    overlap = 0
    if x2-x1 > 0 and y2-y1 > 0 : 
        overlap = (x2-x1) * (y2-y1)
    combined = box1.area + box2.area - overlap
    iou = overlap/combined
    if iou < 0.4 : 
         return 0
    return iou

#create new folder for images
def create_folder(new_path):
    try:
        os.mkdir(new_path)
    except OSError:
        logging.error('Failed to create path' + new_path)
    else:
        logging.info("Succesfully created new path")

# initialize first image with rectangles
def first_image_init(write):
    img = cv2.imread(dict[86])
    for i in img_bounding_boxes[files[85]] :
            # check if height is greater then length -> verify with TA if this is the best way to do this
        new_color = generate_color()
        if write : 
            cv2.rectangle(img,(int(i.x1),int(i.y1+i.h1)), (int(i.x1+i.l1),int(i.y1)), new_color,3)
        color_dict[(86,i.x1,i.y1)] = new_color 
        
    if write : 
        cv2.imwrite('img2/0000'+ str(86) +'.jpg', img)

#find the gt in single image
def find_gt_t(image):
    total = 0
    for i in range(len(rectangle_englobantes_tracking)):
        if rectangle_englobantes_tracking[i][0] == image:
            total += 1
    return total

#MOTA calculation
def calc_mota(fn, fp, ids, gt):
    sum = fn + fp + ids
    return 1 - (sum/gt)


#Find false positives in single image
def find_fp(image):
    fp = 0
    frame_cars = []
    frame_boxes = img_bounding_boxes[files[int(image)-1]]
    for i in range(len(rectangle_englobantes_tracking_voiture)):
        if rectangle_englobantes_tracking_voiture[i][0] == image:
            frame_cars.append(rectangle_englobantes_tracking_voiture[i])
    for i in range(len(frame_boxes)) :
        flag = 0
        for j in range(len(frame_cars)):
            if frame_boxes[i].x1 == int(frame_cars[j][2]) and frame_boxes[i].y1 == int(frame_cars[j][3]) and frame_boxes[i].h1 == int(frame_cars[j][5]) and frame_boxes[i].l1 == int(frame_cars[j][4]):
                flag = 1
        if flag == 0:
            fp+=1
    return fp

#Find false negatives in a single image
def find_fn(image):
    fn = 0
    counter = 0
    frame_cars = []
    frame_boxes = img_bounding_boxes[files[int(image)-1]]
    frame_cars_tuples = []
    frame_boxes_tuples = []
    for i in range(len(rectangle_englobantes_tracking_voiture)):
        if rectangle_englobantes_tracking_voiture[i][0] == image:
            frame_cars.append(rectangle_englobantes_tracking_voiture[i])

    for i in frame_cars :
        frame_cars_tuples.append((int(i[2]),int(i[3]),int(i[4]),int(i[5]))) 


    for i in frame_boxes : 
        frame_boxes_tuples.append((i.x1,i.y1,i.l1,i.h1))

    # Loop to check
    for i in frame_cars_tuples :
        if i not in frame_boxes_tuples : 
            counter += 1
    return counter
            
def calculate_ids(image1,image2) :
    ids = 0
    frame_cars_id_tuples = []
    frame_boxes_id_tuples = []

    frame_cars1 = []
    for i in range(len(rectangle_englobantes_tracking_voiture)):
        if rectangle_englobantes_tracking_voiture[i][0] == image1:
            frame_cars1.append(rectangle_englobantes_tracking_voiture[i])

    frame_cars2 = []
    for i in range(len(rectangle_englobantes_tracking_voiture)):
        if rectangle_englobantes_tracking_voiture[i][0] == image2:
            frame_cars2.append(rectangle_englobantes_tracking_voiture[i])
    
    for i in frame_cars1 : 
        for j in frame_cars2 : 
            if i[1] == j[1] : 
                frame_cars_id_tuples.append((int(i[2]),int(i[3]),int(i[4]),int(i[5]),int(j[2]),int(j[3]),int(j[4]),int(j[5])))

    frame_boxes1 = img_bounding_boxes[files[int(image1)-1]]
    frame_boxes2 = img_bounding_boxes[files[int(image2)-1]]

    for i in frame_boxes1 : 
        flag = False
        for j in frame_boxes2 : 
            if color_dict[(int(image1), i.x1,i.y1)][1] == color_dict[(int(image2), j.x1,j.y1)][1] : 
                frame_boxes_id_tuples.append((i.x1,i.y1,i.l1,i.h1,j.x1,j.y1,j.l1,j.h1))
                flag == True
        if flag == False : 
            frame_boxes_id_tuples.append((i.x1,i.y1,i.l1,i.h1))

    for i in frame_cars_id_tuples : 
        if i not in frame_boxes_id_tuples and ((i[0],i[1],i[2],i[3]) in frame_boxes_id_tuples) : 
            ids += 1
    return ids

#begin iou calculations 
def use_iou(write):
    box_id = 0
    fn = 0
    fp = 0
    ids = 0
    gt_t = 0
    for i in range(85, 467 - 1) :
        
        column = []

        # hold file names for images to examine
        img1 = files[i]
        img2 = files[i + 1]
        img = cv2.imread('img1/'+img2)

        # get bounding box information
        box_list1 = img_bounding_boxes[img1]
        box_list2 = img_bounding_boxes[img2]

        # cycle through bounding boxes for f(x) and f(x + 1)
        for b in range(len(box_list2)) :
            row = []
            for k in range(len(box_list1)) :
                max_index = -1
                iou = calculate_iou(box_list1[k],box_list2[b])
                row.append(iou)
            if len(row) > 0 : 
                max_value = max(row)
                max_index = row.index(max_value)
                row[~max_index] = 0        
                column.append(row)
        if len(column) > 0 : 
            # find max column
            max_column = []
            for j in range(len(box_list1)) :
                max_column.append((0, -1, -1)) 
            for j in range(len(column)) :
                for k in range(len(column[j])) : 
                    val, c, r = max_column[k]
                    if column[j][k] > val : 
                        max_column[k] = (column[j][k],j, k)
            for j in max_column : 
                val, c, r = j
                logging.error((val,c,r,len(column))) 
                column[c][r] = -1
                column[~c][r] = 0
            for j in range(len(column)) : 
                if -1 not in column[j] : 
                    new_color = generate_color()
                    box_id+=1 
                    x1 = box_list2[j].x1
                    y1 = box_list2[j].y1
                    l1 = box_list2[j].l1
                    h1 = box_list2[j].h1
                    if write : 
                        cv2.rectangle(img,(int(x1),int(y1+h1)), (int(x1+l1),int(y1)), new_color,3)
                    color_dict[(i+2,x1,y1)] = [new_color, box_id]
                    with open('new_gt.txt', 'a') as f:
                        f.write(str(i+1) + "," + str(box_id) + "," + str(x1) + "," + str(y1) + "," + str(l1) + "," + str(h1) + "\n") 
                        f.close()         
                else : 
                    x1 = box_list2[j].x1
                    y1 = box_list2[j].y1
                    l1 = box_list2[j].l1
                    h1 = box_list2[j].h1
                    box_type = box_list2[j].box_type

                    orig_index = column[j].index(-1)
                    orig = box_list1[orig_index]

                    new_color = color_dict[(i+1,orig.x1, orig.y1)][0]
                    box_new_id = color_dict[(i+1,orig.x1, orig.y1)][1]
                    if write :    
                        cv2.rectangle(img,(int(x1),int(y1+h1)), (int(x1+l1),int(y1)), new_color,3)
                    color_dict[(i+2,x1,y1)] = [new_color, box_new_id]
                    with open('new_gt.txt', 'a') as f:
                        f.write(str(i+1) + "," + str(box_new_id) + "," + str(x1) + "," + str(y1) + "," + str(l1) + "," + str(h1) + "," +str(box_type) + "\n") 
                        f.close()         

            if write : 
                cv2.imwrite('img2/'+img2, img)
        else :
            if write :  
                cv2.imwrite('img2/'+img2, img)

        # Find sum of false positives, ids, false negatives and total objects
        fp += find_fp(str(i + 2))
        ids += calculate_ids(str(i + 1), str(i + 2))
        gt_t += find_gt_t(str(i + 2))
        fn += find_fn(str(i + 2))
    
    return (fn,fp,ids,gt_t)

#Running all the functions

## Calculating Pedestrian Mota
populate_variables(files)
create_folder("img2")
first_image_init(True)
fn, fp, ids, gt_t = use_iou(True)
print(f"The mota for our algorithm is : \n {calc_mota(fn,fp,ids,gt_t)} \n")


