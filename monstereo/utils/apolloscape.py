
import math

import json



from collections import namedtuple
from .iou import calculate_iou
import numpy as np
#Clusters for apolloscape:
APOLLO_CLUSTERS = ['5', '10', '20', '35', '50', '60', '70', '>74']

#? Set of functions for apolloscape

def euler_angles_to_quaternions(angle):
    """Convert euler angels to quaternions representation.
    Input:
        angle: n x 3 matrix, each row is [roll, pitch, yaw]
    Output:
        q: n x 4 matrix, each row is corresponding quaternion.
    """

    in_dim = np.ndim(angle)
    if in_dim == 1:
        angle = angle[None, :]

    n = angle.shape[0]
    roll, pitch, yaw = angle[:, 0], angle[:, 1], angle[:, 2]
    q = np.zeros((n, 4))

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q[:, 0] = cy * cr * cp + sy * sr * sp
    q[:, 1] = cy * sr * cp - sy * cr * sp
    q[:, 2] = cy * cr * sp + sy * sr * cp
    q[:, 3] = sy * cr * cp - cy * sr * sp

    return q


def intrinsic_mat_to_vec(K_mat):
    """Convert a 3x3 intrinsic vector to a 4 dim intrinsic
       matrix
    """
    return np.array([K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]])


def intrinsic_vec_to_mat(intrinsic, shape=None):
    """Convert a 4 dim intrinsic vector to a 3x3 intrinsic
       matrix
    """
    if shape is None:
        shape = [1, 1]

    K_mat = np.zeros((3, 3), dtype=np.float32)
    K_mat[0, 0] = intrinsic[0] * shape[1]
    K_mat[1, 1] = intrinsic[1] * shape[0]
    K_mat[0, 2] = intrinsic[2] * shape[1]
    K_mat[1, 2] = intrinsic[3] * shape[0]
    K_mat[2, 2] = 1.0

    return K_mat


def round_prop_to(num, base=4.):
    """round a number to integer while being propotion to
       a given base number
    """
    return np.ceil(num / base) * base


def euler_angles_to_rotation_matrix(angle, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = angle[0], angle[1], angle[2]

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = np.array(R)

    if is_dir:
        R = R[:, 2]

    return R


def rotation_matrix_to_euler_angles(R, check=True):
    """Convert rotation matrix to euler angles
    Input:
        R: 3 x 3 rotation matrix
        check: whether Check if a matrix is a valid
            rotation matrix.
    Output:
        euler angle [x/roll, y/pitch, z/yaw]
    """

    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    if check:
        assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])

    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def convert_pose_mat_to_6dof(pose_file_in, pose_file_out):
    """Convert a pose file with 4x4 pose mat to 6 dof [xyz, rot]
    representation.
    Input:
        pose_file_in: a pose file with each line a 4x4 pose mat
        pose_file_out: output file save the converted results
    """

    poses = [line for line in open(pose_file_in)]
    output_motion = np.zeros((len(poses), 6))
    f = open(pose_file_out, 'w')
    for i, line in enumerate(poses):
        nums = line.split(' ')
        mat = [np.float32(num.strip()) for num in nums[:-1]]
        image_name = nums[-1].strip()
        mat = np.array(mat).reshape((4, 4))

        xyz = mat[:3, 3]
        rpy = rotation_matrix_to_euler_angles(mat[:3, :3])
        output_motion = np.hstack((xyz, rpy)).flatten()
        out_str = '%s %s\n' % (image_name, np.array2string(output_motion,
                  separator=',',
                  formatter={'float_kind': lambda x: "%.7f" % x})[1:-1])

        f.write(out_str)
    f.close()

    return output_motion


def trans_vec_to_mat(rot, trans, dim=4):
    """ project vetices based on extrinsic parameters to 3x4 matrix
    """
    mat = euler_angles_to_rotation_matrix(rot)
    mat = np.hstack([mat, trans.reshape((3, 1))])
    if dim == 4:
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])

    return mat


def trans_mat_to_vec(mat):
    rot = rotation_matrix_to_euler_angles(mat[:3, :3])
    trans = mat[:3, 3]
    T = np.array([rot[0],rot[1], rot[2], trans[0], trans[1], trans[2]])

    return T

def project(pose, scale, vertices):
    """ transform the vertices of a 3D car model based on labelled pose
    Input:
        pose: 0-3 rotation, 4-6 translation
        scale: the scale at each axis of the car
        vertices: the vertices position
    """

    if np.ndim(pose) == 1:
        mat = trans_vec_to_mat(pose[:3], pose[3:]) #Obtain the matrix that embed the tranlation and rotation
    elif np.ndim(pose) == 2:
        mat = pose

    vertices = vertices * scale
    p_num = vertices.shape[0]

    points = vertices.copy()
    points = np.hstack([points, np.ones((p_num, 1))])
    points = np.matmul(points, mat.transpose())

    return points[:, :3]

def pose_extraction(model_projected, model):

    p_num =model_projected.shape[0]
    model_projected = np.hstack([model_projected, np.ones((p_num, 1)) ])
    model = np.hstack([model, np.ones((p_num, 1)) ])

    mat = np.matmul(np.linalg.pinv(model), model_projected)

    return trans_mat_to_vec(mat.transpose()), mat.transpose()

def bbox_3d_extract(vertices):

    min_x = np.min(vertices[:,0])
    max_x= np.max(vertices[:,0])
    min_y = np.min(vertices[:,1])
    max_y= np.max(vertices[:,1])
    min_z = np.min(vertices[:,2])
    max_z= np.max(vertices[:,2])

    h = max_y - min_y
    w = max_x - min_x
    l = max_z- min_z

    bbox_3d=[]
    for x in [min_x, max_x]:
        for y in [min_y, max_y]:
            for z in [min_z, max_z]:
                bbox_3d.append([x,y,z])


    return(np.array(bbox_3d), w, l, h)

def car_projection( car_model, scale , T, turn_over = False, bbox= False):
    # perform the projection of the CAD models of the cars in the 3D space (use car_models_json)
    with open(car_model) as json_file:
        data = json.load(json_file) #Loading of the vehicle CAD model
        vertices = np.array(data['vertices']) # extraction of the vertices

        if turn_over: # used to turn the vehicle over its z axis (180 degree turn)
                        # It was shown that the pose estimation have a precomputed
                        # rotation of the carts over their z axis
                        # Reminder: the z axis is the one of the depth
            z_rot = np.array([0.,0., np.pi, 0.0,0.0,0.0])
            vertices = project(z_rot, scale, vertices) #Pre-rotation of the vehicle over its z axis


        vertices_r = project(T, scale, vertices) #projection in the 3D space
        triangles = np.array(data['faces']) - 1  
        results = vertices_r, triangles

        if bbox:
            bbox_3d, w, l, h= bbox_3d_extract(vertices)
            bbox_3d_r = project(T, scale, bbox_3d)      # Place the bounding box at its place in the 3D space
            results = vertices_r, triangles, bbox_3d_r, w, l, h
        else:
            bbox_3d, w, l, h= bbox_3d_extract(vertices)
            results = vertices_r, triangles, bbox_3d, w, l, h
        return results

def pifpaf_info_extractor(json_pifpaf):

    #Extract the list of keypoints from the json files of openpifpaf
    with open(json_pifpaf) as json_file:
        data = json.load(json_file)
        list_keypoints  =[]
        for index, vehicle in enumerate(data):
            keypoints = np.reshape(vehicle['keypoints'], [-1,3])[:,[2,0,1]]
            bbox = vehicle['bbox']
            score = vehicle['score']
            cat_id = vehicle['category_id']

            list_keypoints.append(keypoints)

        return list_keypoints

def keypoint_expander(vertices_2d, keypoints, buffer = 30, kps_3d = True) :
    #process the existing keypoints and assign them a depth by minimizing the
    #distance with the vertices of the projected CAD models.

    new_keypoints =[]
    if len(keypoints.shape) == 1:
        keypoints = [keypoints]

    if kps_3d == False:
        return np.array(keypoints)

    for keypoint in keypoints:
        res = np.ones([len(vertices_2d),2])*keypoint[1:] - np.transpose([vertices_2d[:,0]/vertices_2d[:,2],
                                                                        vertices_2d[:,1]/vertices_2d[:,2]])

        dist = np.linalg.norm(res, axis = 1)

        z = np.min(vertices_2d[np.argsort(dist)[0:buffer], 2])


        if keypoint[0]==0:
            new_keypoints.append([keypoint[0], keypoint[1], keypoint[2], -5])
        else:
            new_keypoints.append([keypoint[0], keypoint[1], keypoint[2], z])

    return np.array(new_keypoints)

def keypoint_projection(keypoints_3D_img, intrinsic_matrix) :
        #take the keypoints in the 2D space form the ground truth and re-project them in the 3D space.
    intrinsic_matrix = np.array(intrinsic_matrix)
    keypoints_3D_img = np.array(keypoints_3D_img)
    keypoints_3D_img[:,1] = keypoints_3D_img[:,1] *keypoints_3D_img[:,3]
    keypoints_3D_img[:,2] = keypoints_3D_img[:,2] *keypoints_3D_img[:,3]
    keypoints_3D = np.matmul(np.array(keypoints_3D_img)[:,1:],
                            np.linalg.pinv(intrinsic_matrix.transpose()))   #The first element is the 
                                                                            #index of the keypoints
    return np.array( keypoints_3D)


def get_iou_matches_custom(boxes, boxes_gt, iou_min=0.3):
    """From 2 sets of boxes and a minimum threshold, compute the matching indices for IoU matches"""

    matches = []
    used = []
    if not boxes or not boxes_gt:
        return []
    confs = [box[4] for box in boxes]

    indices = list(np.argsort(confs))
    for idx in indices[::-1]:
        box = boxes[idx]
        ious = []
        for idx_gt, box_gt in enumerate(boxes_gt):
            iou = calculate_iou(box, box_gt)
            ious.append(iou)
        idx_gt_max = int(np.argmax(ious))
        if (ious[idx_gt_max] >= iou_min) and (idx_gt_max not in used):
            matches.append((idx, idx_gt_max))
            used.append(idx_gt_max)
    return matches, ious

def keypoints_to_cad_model(keypoints, vertices_cad_dic, iou = 0.3):
    


    kps = keypoints[keypoints[:,0]>0][:, 1:]
    conf = np.sum(keypoints[:,0])
    box_kp = [min(kps[:,0]), min(kps[:,1]),max(kps[:,0]), max(kps[:,1]), conf ]

    boxes_gt = []
    for i, vertices_2d in vertices_cad_dic.items():
        x,y = (vertices_2d[:,0]/vertices_2d[:,2], vertices_2d[:,1]/vertices_2d[:,2])
        boxes_gt.append([min(x),min(y), max(x), max(y)])

    matches, ious = get_iou_matches_custom([box_kp], boxes_gt, iou_min=iou)

    if len(matches) == 0:
        return None, 0.0
    else:
        return matches[0][1], np.max(ious)


if __name__ == '__main__':
    pass


"""
    Brief: Car model summary
"""

#Camera parameters extracted from the dataset

K ={
            'Camera_5': np.array(
                [2304.54786556982, 2305.875668062,
                 1686.23787612802, 1354.98486439791]),
            'Camera_6': np.array(
                [2300.39065314361, 2301.31478860597,
                 1713.21615190657, 1342.91100799715])}

K_ext= {
            'R': np.array([
                [9.96978057e-01, 3.91718762e-02, -6.70849865e-02],
                [-3.93257593e-02, 9.99225970e-01, -9.74686202e-04],
                [6.69948100e-02, 3.60985263e-03, 9.97746748e-01]]),
            'T': np.array([-0.6213358, 0.02198739, -0.01986043])
        }


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------


#Keypoints id used in this following order for openpifpaf
#KPS_MAPPING = [49, 8, 57, 0, 52, 5, 11, 7, 20, 23, 24, 33, 25, 32, 28, 29, 46, 34, 37, 50]
KPS_MAPPING = [49, 8, 57, 0, 52, 5, 11, 7, 20, 23, 24, 33, 25, 32, 28, 29, 46, 34, 37, 50, 65, 64, 9, 48]


# a label and all meta information
Label = namedtuple('Label', [

    'name'        , # The name of a car type
    'id'          , # id for specific car type
    'category'    , # The name of the car category, 'SUV', 'Sedan' etc
    'categoryId'  , # The ID of car category. Used to create ground truth images
                    # on category level.
    ])

# Thresholds for the evaluation
#threshold_formatting =[threshold shape, threshold_trans_abs, threshold_rot, threshold_trans_rel]
threshold_loose = [0.5 , 2.8 , np.pi/6, 0.1]
threshold_strict = [0.75, 1.4 , np.pi/12, 0.05]
threshold_mean = np.mean([[0.5, 2.8, np.pi/6, 0.1], [0.95, 0.1, np.pi/60, 0.01]], axis = 0)


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

models = [
    #     name          id   is_valid  category  categoryId
    Label(             'baojun-310-2017',          0,       '2x',          0),
    Label(                'biaozhi-3008',          1,       '2x',          0),
    Label(          'biaozhi-liangxiang',          2,       '2x',          0),
    Label(           'bieke-yinglang-XT',          3,       '2x',          0),
    Label(                'biyadi-2x-F0',          4,       '2x',          0),
    Label(               'changanbenben',          5,       '2x',          0),
    Label(                'dongfeng-DS5',          6,       '2x',          0),
    Label(                     'feiyate',          7,       '2x',          0),
    Label(         'fengtian-liangxiang',          8,       '2x',          0),
    Label(                'fengtian-MPV',          9,       '2x',          0),
    Label(           'jilixiongmao-2015',         10,       '2x',          0),
    Label(           'lingmu-aotuo-2009',         11,       '2x',          0),
    Label(                'lingmu-swift',         12,       '2x',          0),
    Label(             'lingmu-SX4-2012',         13,       '2x',          0),
    Label(              'sikeda-jingrui',         14,       '2x',          0),
    Label(        'fengtian-weichi-2006',         15,       '3x',          1),
    Label(                   '037-CAR02',         16,       '3x',          1),
    Label(                     'aodi-a6',         17,       '3x',          1),
    Label(                   'baoma-330',         18,       '3x',          1),
    Label(                   'baoma-530',         19,       '3x',          1),
    Label(            'baoshijie-paoche',         20,       '3x',          1),
    Label(             'bentian-fengfan',         21,       '3x',          1),
    Label(                 'biaozhi-408',         22,       '3x',          1),
    Label(                 'biaozhi-508',         23,       '3x',          1),
    Label(                'bieke-kaiyue',         24,       '3x',          1),
    Label(                        'fute',         25,       '3x',          1),
    Label(                     'haima-3',         26,       '3x',          1),
    Label(               'kaidilake-CTS',         27,       '3x',          1),
    Label(                   'leikesasi',         28,       '3x',          1),
    Label(               'mazida-6-2015',         29,       '3x',          1),
    Label(                  'MG-GT-2015',         30,       '3x',          1),
    Label(                       'oubao',         31,       '3x',          1),
    Label(                        'qiya',         32,       '3x',          1),
    Label(                 'rongwei-750',         33,       '3x',          1),
    Label(                  'supai-2016',         34,       '3x',          1),
    Label(             'xiandai-suonata',         35,       '3x',          1),
    Label(            'yiqi-benteng-b50',         36,       '3x',          1),
    Label(                       'bieke',         37,       '3x',          1),
    Label(                   'biyadi-F3',         38,       '3x',          1),
    Label(                  'biyadi-qin',         39,       '3x',          1),
    Label(                     'dazhong',         40,       '3x',          1),
    Label(              'dazhongmaiteng',         41,       '3x',          1),
    Label(                    'dihao-EV',         42,       '3x',          1),
    Label(      'dongfeng-xuetielong-C6',         43,       '3x',          1),
    Label(     'dongnan-V3-lingyue-2011',         44,       '3x',          1),
    Label(    'dongfeng-yulong-naruijie',         45,      'SUV',          2),
    Label(                     '019-SUV',         46,      'SUV',          2),
    Label(                   '036-CAR01',         47,      'SUV',          2),
    Label(                 'aodi-Q7-SUV',         48,      'SUV',          2),
    Label(                  'baojun-510',         49,      'SUV',          2),
    Label(                    'baoma-X5',         50,      'SUV',          2),
    Label(             'baoshijie-kayan',         51,      'SUV',          2),
    Label(             'beiqi-huansu-H3',         52,      'SUV',          2),
    Label(              'benchi-GLK-300',         53,      'SUV',          2),
    Label(                'benchi-ML500',         54,      'SUV',          2),
    Label(         'fengtian-puladuo-06',         55,      'SUV',          2),
    Label(            'fengtian-SUV-gai',         56,      'SUV',          2),
    Label(    'guangqi-chuanqi-GS4-2015',         57,      'SUV',          2),
    Label(        'jianghuai-ruifeng-S3',         58,      'SUV',          2),
    Label(                  'jili-boyue',         59,      'SUV',          2),
    Label(                      'jipu-3',         60,      'SUV',          2),
    Label(                  'linken-SUV',         61,      'SUV',          2),
    Label(                   'lufeng-X8',         62,      'SUV',          2),
    Label(                 'qirui-ruihu',         63,      'SUV',          2),
    Label(                 'rongwei-RX5',         64,      'SUV',          2),
    Label(             'sanling-oulande',         65,      'SUV',          2),
    Label(                  'sikeda-SUV',         66,      'SUV',          2),
    Label(            'Skoda_Fabia-2011',         67,      'SUV',          2),
    Label(            'xiandai-i25-2016',         68,      'SUV',          2),
    Label(            'yingfeinidi-qx80',         69,      'SUV',          2),
    Label(             'yingfeinidi-SUV',         70,      'SUV',          2),
    Label(                  'benchi-SUR',         71,      'SUV',          2),
    Label(                 'biyadi-tang',         72,      'SUV',          2),
    Label(           'changan-CS35-2012',         73,      'SUV',          2),
    Label(                 'changan-cs5',         74,      'SUV',          2),
    Label(          'changcheng-H6-2016',         75,      'SUV',          2),
    Label(                 'dazhong-SUV',         76,      'SUV',          2),
    Label(     'dongfeng-fengguang-S560',         77,      'SUV',          2),
    Label(       'dongfeng-fengxing-SX6',         78,      'SUV',          2)

]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
car_name2id = {label.name: label for label in models}
car_id2name = {label.id: label for label in models}

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

