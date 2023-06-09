import os
import sys
import time
import copy
import json
import logging
from collections import defaultdict
import datetime
import numpy as np
import torch
from ..utils import correct_angle, to_spherical #,append_cluster

from ..network.process import preprocess_monoloco, keypoints_dropout, clear_keypoints

from ..utils import K , APOLLO_CLUSTERS ,car_id2name, intrinsic_vec_to_mat,car_projection,\
                    pifpaf_info_extractor, keypoint_expander, \
                    set_logger, get_iou_matches




class PreprocessApolloscape:


    """Preprocess apolloscape dataset"""


    dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                            clst=defaultdict(lambda: defaultdict(list))),
              'val': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                          clst=defaultdict(lambda: defaultdict(list))),
                          }
    dic_names = defaultdict(lambda: defaultdict(list))


    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def __init__(self, dir_ann, dataset, kps_3d = False, buffer=30,  dropout = 0, confidence = False, iou_min = 0.3, transformer = False):

        logging.basicConfig(level=logging.INFO)

        self.buffer = buffer

        self.transformer = transformer

        self.dropout =dropout
        
        self.kps_3d = kps_3d
        
        self.dir_ann = dir_ann

        self.confidence = confidence

        self.iou_min = iou_min

        dir_apollo = os.path.join('data', 'apolloscape')
        dir_out = os.path.join('data', 'arrays')

        assert os.path.exists(dir_apollo), "apollo directory does not exists"
        assert os.path.exists(self.dir_ann), "The annotations directory does not exists"
        assert os.path.exists(dir_out), "Joints directory does not exists"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M%S")[2:]


        try:
            process_mode = os.environ["process_mode"]
        except:
            process_mode = "NULL"

        identifier = '-apolloscape'

        kps_3d_id =""
        if self.kps_3d:
            identifier+="-kps_3d"
            kps_3d_id+="-kps_3d"

        if self.transformer:
            identifier+="-transformer"
            kps_3d_id+="-transformer"

        name_out = 'ms-' + now_time + identifier+"-prep"+".txt"

        self.logger = set_logger(os.path.join('data', 'logs', name_out))
        self.logger.info("Preparation arguments: \nDir_ann: {} "
                         "\nprocess_mode : {} \nDropout images: {} \nConfidence keypoints:"
                         " {} \nKeypoints 3D: {} \nTransformer: {}".format(dir_ann, process_mode, dropout, 
                                                                            confidence, self.kps_3d, self.transformer))


        self.path_joints = os.path.join(dir_out, 'joints-apolloscape-' + dataset + kps_3d_id + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-apolloscape-' + dataset + kps_3d_id + '-' + now_time + '.json')

        
        self.path  = os.path.join(dir_apollo, dataset)
        self.scenes, self.train_scenes, self.validation_scenes = factory(dataset, dir_apollo)



        
    def run(self):
        """
        Prepare arrays for training
        """
        cnt_scenes  = cnt_ann = 0
        start = time.time()

        occluded_keypoints=defaultdict(list)
        
        #? In case of a dropout, the processing is rolled two times
        #? the first time with a dropout of 0 and then with a dropout on the key-points
        #? equal to self.dropout
        if self.dropout>0:
            dropouts = [0, self.dropout]
        else:
            dropouts = [0]

        for dropout in dropouts:
            #? inner loop for each dropout
            
            if len(dropouts)>=2:
                self.logger.info("Generation of the inputs for a dropout of {}".format(dropout))

            for ii, scene in enumerate(self.scenes):
                
                cnt_scenes +=1 
                
                scene_id = scene.split("/")[-1].split(".")[0]
                camera_id = scene_id.split("_")[-1]
                car_poses = os.path.join(self.path, "car_poses", scene_id + ".json")
                
                #print(scene_id, self.train_scenes[:5])
                if scene_id+".jpg" in self.train_scenes:
                    phase = 'train'
                elif scene_id+".jpg" in self.validation_scenes:
                    phase ='val'
                else:    
                    print("phase name not in training or validation split")
                    continue
                    
                kk = K["Camera_"+camera_id]
                
                path_im = scene
                
                # Run IoU with pifpaf detections and save
                path_pif = os.path.join(self.dir_ann, scene_id+".jpg" + '.predictions.json')
                            
                if os.path.isfile(path_pif):
                    #? extract the required inputs/outputs from the ground truth and pifpaf predictions
                    boxes_gt_list, boxes_3d_list, kps_list, \
                    ys_list, car_model_list  = self.extract_ground_truth_pifpaf(car_poses,camera_id, scene_id, path_pif)

                else:
                    raise ValueError("Please, provide the right pifpaf annotations for the annotations" 
                                    "(in case you are using apolloscape mini, please preprocess the images first)")

                if dropout == 0: 
                    #? For the manual evaluation, the extended dataset with the dropout is not considered.
                    # We train on the extended dataset and evaluate on the original dataset
                    self.dic_names[scene_id+".jpg"]['boxes'] = copy.deepcopy(list(boxes_gt_list))
                    self.dic_names[scene_id+".jpg"]['car_model'] = copy.deepcopy(car_model_list)

                    self.dic_names[scene_id+".jpg"]['K'] = copy.deepcopy(intrinsic_vec_to_mat(kk).tolist())
                    ys_list_final = []
                

                if phase == 'val' and dropout > 0.0:
                    #? If we are in the validation case, we should not use the dropout parameter.
                    continue
                
                

                for kps, ys, boxes_gt, boxes_3d in zip(kps_list, ys_list, boxes_gt_list, boxes_3d_list):
                    
                    kps = [kps.transpose().tolist()]                    
                    
                    #? perform the dropout on the key-points
                    kps, length_keypoints, occ_kps = keypoints_dropout(kps, dropout, kps_3d = self.kps_3d)
                    occluded_keypoints[phase].append(occ_kps)
                    #? project the key-points from the pixel frame to the image frame
                    inp = preprocess_monoloco(kps,  intrinsic_vec_to_mat(kk).tolist(), 
                                            confidence =self.confidence).view(-1).tolist()
                    
                    if self.kps_3d:
                        #? in case of the kps_3d option, the depth coordinates are 
                        #? added at the end of the ys vector for the prediction part
                        keypoints = clear_keypoints(kps, 3)
                        z = (keypoints[:, 2, :]).tolist()[0]
                        ys = list(ys)
                        ys.append(z)
                        ys_list_final.append(ys)

                    self.dic_jo[phase]['kps'].append(kps.tolist())
                    self.dic_jo[phase]['X'].append(list(inp))
                    self.dic_jo[phase]['Y'].append(ys)

                    #? Mark the different types of dropout instanciated -> Data augmentation
                    if dropout!=0:
                        mark = "_drop_0"+str(dropout).split(".")[-1]
                    else:
                        mark =''
                    self.dic_jo[phase]['names'].append(scene_id+mark+".jpg")  # One image name for each annotation
                    self.dic_jo[phase]['boxes_3d'].append(list(boxes_3d))
                    self.dic_jo[phase]['K'].append(intrinsic_vec_to_mat(kk).tolist())
                    
                    append_cluster(self.dic_jo, phase, list(inp), ys, kps.tolist())
                    cnt_ann += 1
                    sys.stdout.write('\r' + 'Saved annotations {}'.format(cnt_ann) + '\t')

                if dropout == 0:
                    #? For the manual evaluation, the extended dataset with the dropout is not considered.
                    #? We train on the extended dataset and evaluate on the original dataset

                    #self.dic_names[scene_id+".jpg"]['ys'] = copy.deepcopy(ys_list_final if self.kps_3d else ys_list)
                    self.dic_names[scene_id+".jpg"]['ys'] = copy.deepcopy( ys_list)


        with open(os.path.join(self.path_joints), 'w') as f:
            json.dump(self.dic_jo, f)
        with open(os.path.join(self.path_names), 'w') as f:
            json.dump(self.dic_names, f)
        end = time.time()

        extract_box_average(self.dic_jo['train']['boxes_3d'])
        print("\nSaved {} annotations for {} scenes. Total time: {:.1f} minutes".format(cnt_ann, cnt_scenes, (end-start)/60))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))    


        mean_val =   torch.mean(torch.Tensor(occluded_keypoints['val']))
        mean_train = torch.mean(torch.Tensor(occluded_keypoints['train']))
        std_val =    torch.std(torch.Tensor(occluded_keypoints['val']))
        std_train=   torch.std(torch.Tensor(occluded_keypoints['train']))

        self.logger.info("\nNumber of keypoints in the skeleton : {}\n"
                          "Val: mean occluded keypoints {:.4}; STD {:.4}\n"
                          "Train: mean occluded keypoints {:.4}; STD {:.4}\n".format(length_keypoints, 
                                                                                    mean_val, std_val, mean_train, std_train) )
        self.logger.info("\nOutput files:\n{}\n{}".format(self.path_names, self.path_joints))
        self.logger.info('-' * 120)
                 
    def extract_ground_truth_pifpaf(self, car_poses,camera_id, scene_id, path_pif):
        """
        Extract the ground truth of apolloscape and reformat it in our desired formatting.
        Additionally, the predicted pifpaf keypoints are used with the ground truth CAD models
        to estimate their depth.
        This is useful for the estimation of the depth component of the keypoints triggered by --kps_3d
        """


        with open(car_poses) as json_file:
            data = json.load(json_file) #open the pose of the cars
        dic_vertices = {}
        dic_boxes = {}
        dic_poses = {}
        vertices_to_keypoints = {}
        dic_keypoints = {}
        dic_car_model = {}

        #Extract the boxes, vertices and the poses of each cars                   
        for car_index, car in enumerate(data):
            name_car = car_id2name[car['car_id']].name 
            car_model = os.path.join(self.path, "car_models_json",name_car+".json")

            intrinsic_matrix = intrinsic_vec_to_mat(K["Camera_"+camera_id]) # reformat the intrinsic matrix

            #? extract the 3D localization from the pose of each vehicle
            vertices_r, triangles, _ , w, l, h = car_projection(car_model, np.array([1,1,1]),
                                                                 T = np.array(car['pose']),  turn_over = True, bbox = False)
            
            #? project the 3D vertices of the CAD model to the 2D space
            vertices_2d = np.matmul(vertices_r,intrinsic_matrix.transpose()) # Projected vertices on the 2D plane
            x,y = (vertices_2d[:,0]/vertices_2d[:,2], vertices_2d[:,1]/vertices_2d[:,2])
            box_gt = [np.min(x), np.min(y), np.max(x), np.max(y)]
            
            dic_vertices[car_index] = vertices_2d
            dic_boxes[car_index] = [box_gt, w, l, h]
            dic_poses[car_index] = np.array(car['pose'])
            dic_car_model[car_index] = car_model

        new_keypoints = None
        
        keypoints_list = []
        boxes_gt_list = []  # Countain the the 2D bounding box of the vehicles
        boxes_3d_list = []
        ys_list = []     
        car_model_list = []          
        
        keypoints_pifpaf = pifpaf_info_extractor(path_pif)


        #? Match the 2d keypoints of the dataset with the keypoints from pifpaf
        boxes_kps=[]
        for index_keypoints, keypoints in enumerate(keypoints_pifpaf):
            
            dic_keypoints[index_keypoints] = np.array(keypoints)
            kps = keypoints[keypoints[:,0]>0][:, 1:]
            if len(kps)>1:
                conf = np.sum(keypoints[:,0])
                box_kp = [min(kps[:,0]), min(kps[:,1]),max(kps[:,0]), max(kps[:,1]), conf ]
                boxes_kps.append(box_kp)

        boxes_gt=[]
        for i, vertices_2d in dic_vertices.items():
            x,y = (vertices_2d[:,0]/vertices_2d[:,2], vertices_2d[:,1]/vertices_2d[:,2])
            boxes_gt.append([min(x),min(y), max(x), max(y)])
            
        matches = get_iou_matches(boxes_kps, boxes_gt, iou_min=self.iou_min)

        for match in matches:
            index_keypoint = match[0]
            index_vertice = match[1]
            
            vertices_to_keypoints[index_vertice] = [index_keypoint, None]
        

        #? With the keypoints now matched, extract the depth of the key-points from the 
        #? ground-truth and apply them to the keypoints
        for index_cad, (index_keypoints, count) in vertices_to_keypoints.items()   :

            keypoints = dic_keypoints[index_keypoints]
            vertices_2d = dic_vertices[index_cad]
            
            new_keypoints = keypoint_expander(vertices_2d, keypoints, self.buffer ,self.kps_3d)

            if self.kps_3d :
                keypoints_list.append( new_keypoints[:,[1,2,3,0]])
            else:
                keypoints_list.append( new_keypoints[:,[1,2,0]])
                                
            boxes_gt_list.append(dic_boxes[index_cad][0]) #2D corners of the bounding box
            
            
            w, l, h = dic_boxes[index_cad][1:]
            roll, pitch, yaw, xc, yc, zc= dic_poses[index_cad] # Center position of the car and its orientation
            boxes_3d_list.append([xc, yc, zc, w, l, h])
            yaw = yaw%(np.pi*2)
            if yaw > np.pi:
                yaw = yaw - 2*np.pi
            elif yaw < -np.pi:
                yaw = yaw + np.pi*2
            sin, cos, _ = correct_angle(yaw, [xc, yc, zc])
            
            if True :
                rtp = to_spherical([xc, yc, zc])
                r, theta, psi = rtp # With r =d = np.linalg.norm([xc,yc,zc]) -> conversion to spherical coordinates 
                ys_list.append([theta, psi, zc, r, h, w, l, sin, cos, yaw])
            else:
                ys_list.append([xc, yc, zc, np.linalg.norm([xc, yc, zc]), h, w, l, sin, cos, yaw])
            
            
            car_model_list.append(dic_car_model[index_cad])
            
        return boxes_gt_list, boxes_3d_list, keypoints_list, ys_list, car_model_list

def factory(dataset, dir_apollo):
    """Define dataset type and split training and validation"""

    assert dataset in ['train', '3d_car_instance_sample']

    path = os.path.join(dir_apollo, dataset)
    if "sample" in dataset:
        path = os.path.join(path, '3d_car_instance_sample')
    
    if dataset == 'train':
        
        with open(os.path.join(path, "split", "train-list.txt"), "r") as file:
            train_scenes = file.read().splitlines()
        with open(os.path.join(path, "split", "validation-list.txt"), "r") as file:
            validation_scenes = file.read().splitlines()
            
    elif dataset == '3d_car_instance_sample':
        with open(os.path.join(path,"split", "train.txt"), "r") as file:
            train_scenes = file.read().splitlines()
        with open(os.path.join(path,  "split", "val.txt"), "r") as file:
            validation_scenes = file.read().splitlines()
    
    path_img = os.path.join(path, "images")
    scenes = [os.path.join(path_img, file) for file in os.listdir(path_img) if file.endswith(".jpg")]
    
    return scenes, train_scenes, validation_scenes



def extract_box_average(boxes_3d):
    boxes_np = np.array(boxes_3d)
    means = np.mean(boxes_np[:, 3:], axis=0)
    stds = np.std(boxes_np[:, 3:], axis=0)





def bbox_gt_extract(bbox_3d, kk):
    zc = np.mean(bbox_3d[:,2])
    
    #?take the top right corner and the bottom left corner of the bounding box in the 3D space
    corners_3d = np.array([[np.min(bbox_3d[:,0]), np.min(bbox_3d[:,1]), zc], [np.max(bbox_3d[:,0]), np.max(bbox_3d[:,1]), zc] ])
    
    box_2d = []
    
    for xyz in corners_3d:
        xx, yy, zz = np.dot(kk, xyz)
        uu = xx / zz
        vv = yy / zz
        box_2d.append(uu)
        box_2d.append(vv)

    return box_2d
    
def append_cluster(dic_jo, phase, xx, ys, kps):
    """Append the annotation based on its distance"""

    for clst in APOLLO_CLUSTERS:
        try:
            if ys[3] <= int(clst):
                dic_jo[phase]['clst'][clst]['kps'].append(kps)
                dic_jo[phase]['clst'][clst]['X'].append(xx)
                dic_jo[phase]['clst'][clst]['Y'].append(ys)
                break
            
        except ValueError:
            dic_jo[phase]['clst'][clst]['kps'].append(kps)
            dic_jo[phase]['clst'][clst]['X'].append(xx)
            dic_jo[phase]['clst'][clst]['Y'].append(ys)
