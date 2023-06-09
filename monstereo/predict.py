
# pylint: disable=too-many-statements, too-many-branches, undefined-loop-variable

import os
import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from .visuals.printer import Printer
from .visuals.pifpaf_show import KeypointPainter, image_canvas
#from .network import PifPaf
from .network import  ImageList, Loco
from .network.process import factory_for_gt, preprocess_pifpaf

from .utils import open_annotations

def predict(args):

    cnt = 0

    # Load Models
    #? NOT COMPATIBLE WITH THE LATEST OPENPIFPAF VERSION
    #pifpaf = PifPaf(args)
    assert args.mode in ('mono', 'stereo', 'pifpaf')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'mono' in args.mode:
        monoloco = Loco(model=args.model, net='monoloco_pp',
                        device=device, n_dropout=args.n_dropout, p_dropout=args.dropout, vehicles = args.vehicles,
                        kps_3d=args.kps_3d, confidence=args.confidence,transformer = args.transformer,
                        lstm = args.lstm, scene_disp = args.scene_disp)

    if 'stereo' in args.mode:
        monstereo = Loco(model=args.model, net='monstereo',
                        device=device, n_dropout=args.n_dropout, p_dropout=args.dropout, vehicles = args.vehicles,
                        kps_3d=args.kps_3d, confidence = args.confidence, transformer = args.transformer,
                        lstm = args.lstm, scene_disp = args.scene_disp)


    # data
    data = ImageList(args.images, scale=args.scale)
    if args.mode == 'stereo':
        assert len(data.image_paths) % 2 == 0, "Odd number of images in a stereo setting"
        bs = 2
    else:
        bs = 1

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=bs, shuffle=False)#,
        #pin_memory=args.pin_memory, num_workers=args.loader_workers)

    for idx, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):
        images = image_tensors.permute(0, 2, 3, 1)


        if not args.joints_folder is None:
            processed_images = processed_images_cpu.to(device, non_blocking=True)
            #fields_batch = pifpaf.fields(processed_images)     #! Waiting for the new integratio n with pifpaf
            fields_batch = image_paths

        # unbatch stereo pair
        for ii, (image_path, image, processed_image_cpu, fields) in enumerate(zip(
                image_paths, images, processed_images_cpu, fields_batch)):


            if not args.joints_folder is None:
                assert os.path.isdir(args.joints_folder), 'invalid path dir'
                img_id = image_path.split("/")[-1].split(".")[0]
                img_type = image_path.split("/")[-1].split(".")[1]


                if ii == 0:

                    if img_id + "." +img_type+".predictions.json" in os.listdir(args.joints_folder):
                        joints_path = os.path.join(args.joints_folder, img_id + "."+img_type+".predictions.json")
                        pifpaf_out = open_annotations(joints_path)

                        pifpaf_outputs = [None, None, pifpaf_out]  # keypoints_sets and scores for pifpaf printing
                        images_outputs = [image]    #List of 1 or 2 elements with pifpaf tensor and monoloco
                                                    #original image
                        pifpaf_outs = {'left': pifpaf_out}
                        image_path_l = image_path
                        if args.output_directory is None:
                            output_path = image_paths[0]
                        else:
                            file_name = os.path.basename(image_paths[0])
                            output_path = os.path.join(args.output_directory, file_name)
                else:
                    if img_id + "." +img_type+".predictions.json" in os.listdir(args.joints_folder+'_right'):
                        joints_path = os.path.join(args.joints_folder+'_right',
                                                    img_id + "."+img_type+".predictions.json")
                        pifpaf_out = open_annotations(joints_path)
                        pifpaf_outs['right'] = pifpaf_out
            else:
                if args.output_directory is None:
                    output_path = image_paths[0]
                else:
                    file_name = os.path.basename(image_paths[0])
                    output_path = os.path.join(args.output_directory, file_name)

                #! NOT COMPATIBLE WITH THE LATEST OPENPIFPAF VERSION
                keypoint_sets, scores, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)
                if ii == 0:
                    pifpaf_outputs = [keypoint_sets, scores, pifpaf_out]    #keypoints_sets and scores for pifpaf
                                                                            #printing
                    images_outputs = [image]    # List of 1 or 2 elements with pifpaf
                                                #tensor and monoloco original image
                    pifpaf_outs = {'left': pifpaf_out}
                    image_path_l = image_path
                else:
                    pifpaf_outs['right'] = pifpaf_out

        if args.mode in ('stereo', 'mono'):
            # Extract calibration matrix and ground truth file if present
            with open(image_path_l, 'rb') as f:
                pil_image = Image.open(f).convert('RGB')
                images_outputs.append(pil_image)

            im_name = os.path.basename(image_path_l)
            im_size = (float(image.size()[1] / args.scale), float(image.size()[0] / args.scale))  # Original
            kk, dic_gt = factory_for_gt(im_size, name=im_name, path_gt=args.path_gt)
            # Preprocess pifpaf outputs and run monoloco
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['left'], im_size, enlarge_boxes=False)
            if args.mode == 'mono':
                print("Prediction with MonoLoco++")
                dic_out = monoloco.forward(keypoints, kk)
                dic_out = monoloco.post_process(dic_out, boxes, keypoints, kk, dic_gt, kps_3d=args.kps_3d)
                #print("RESULTS", dic_out)
            else:
                print("Prediction with MonStereo")
                boxes_r, keypoints_r = preprocess_pifpaf(pifpaf_outs['right'], im_size)
                dic_out = monstereo.forward(keypoints, kk, keypoints_r=keypoints_r)
                dic_out = monstereo.post_process(dic_out, boxes, keypoints, kk, dic_gt, kps_3d = args.kps_3d)

        else:
            dic_out = defaultdict(list)
            kk = None

        factory_outputs(args, images_outputs, output_path, pifpaf_outputs, dic_out=dic_out,
                        kk=kk, vehicles=args.vehicles)
        print('Image {}\n'.format(cnt) + '-' * 120)
        cnt += 1


def factory_outputs(args, images_outputs, output_path, pifpaf_outputs, dic_out=None, kk=None, vehicles = False):
    """Output json files or images according to the choice"""

    # Save json file
    if args.mode == 'pifpaf':
        keypoint_sets, scores, pifpaf_out = pifpaf_outputs[:]


        if not args.joints_folder is None:
            keypoint_sets = []
            for pifpaf_o in pifpaf_out:
                keypoint_sets.append(np.reshape(pifpaf_o['keypoints'], (3,-1)))

        CAR_SKELETON = [[1, 17], [1, 2], [1, 3], [2, 4], [2, 7], [3, 4], [3, 5],
            [4, 6], [5, 6], [6, 8], [8, 9], [7, 10], [9, 10], [10, 13], [18, 14],
            [23, 7], [23, 8], [23, 9], [8, 4], [23, 4], [23, 2], [11, 12], [11, 13],
            [11, 7], [13, 14], [13, 15], [12, 14], [12, 17], [14, 16], [15, 16], [21, 22],
            [21, 13], [21, 15], [21, 9], [22, 16], [22, 14], [22, 19], [16, 19], [15, 9], [18, 19],
            [18, 17], [19, 20], [20, 5], [17, 24], [24, 19], [24, 1], [20, 3], [24, 3], [24, 20]]

        if vehicles:
            skeleton = CAR_SKELETON
        else:
            skeleton = False
        # Visualizer
        keypoint_painter = KeypointPainter(show_box=False, skeleton =skeleton)
        skeleton_painter = KeypointPainter(show_box=False, color_connections=True, markersize=1,
                                            linewidth=4, skeleton = skeleton)

        if 'json' in args.output_types and pifpaf_out.size > 0:
            with open(output_path + '.pifpaf.json', 'w') as f:
                json.dump(pifpaf_out, f)

        if 'keypoints' in args.output_types:
            with image_canvas(images_outputs[0],
                              output_path + '.keypoints.png',
                              show=args.show,
                              fig_width=args.figure_width,
                              dpi_factor=args.dpi_factor) as ax:
                keypoint_painter.keypoints(ax, keypoint_sets)

        if 'skeleton' in args.output_types:
            with image_canvas(images_outputs[0],
                              output_path + '.skeleton.png',
                              show=args.show,
                              fig_width=args.figure_width,
                              dpi_factor=args.dpi_factor) as ax:
                skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)

    else:
        if any((xx in args.output_types for xx in ['front', 'bird', 'combined', 'combined_3d',
                                                    'combined_kps', 'combined_nkps', '3d_visu'])):

            epistemic = False
            if args.n_dropout > 0:
                epistemic = True

            if dic_out['boxes']:  # Only print in case of detections
                printer = Printer(images_outputs[1], output_path, kk, output_types=args.output_types,
                                z_max=args.z_max, epistemic=epistemic)
                figures, axes = printer.factory_axes()

                if False :
                    #? return a white background as the image
                    im = images_outputs[1].convert('RGBA')
                    data = np.array(im)
                    print("HERE", data.shape)
                    data[..., :-1] = (255,255,255)
                    images_outputs[1]  = Image.fromarray(data)

                printer.draw(figures, axes, dic_out, images_outputs[1], show_all=args.show_all,
                            draw_box=args.draw_box, save=True, show=args.show, kps = pifpaf_outputs)

        if 'json' in args.output_types:
            with open(os.path.join(output_path + '.monoloco.json'), 'w') as ff:
                json.dump(dic_out, ff)
