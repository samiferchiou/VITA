"""
Evaluate MonStereo code on KITTI dataset using ALE metric
"""

# pylint: disable=attribute-defined-outside-init

import os
import math
import logging
import datetime
from collections import defaultdict
import numpy as np

from tabulate import tabulate

from ..utils import get_iou_matches, get_task_error, get_pixel_error, check_conditions, \
    get_difficulty, split_training, parse_ground_truth, get_iou_matches_matrix, set_logger
from ..visuals import show_results, show_spread, show_task_error, show_box_plot


class EvalKitti:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    CLUSTERS = ('easy', 'moderate', 'hard', 'all', '3', '5', '7', '9', '11', '13', '15', '17', '19', '21', '23', '25',
                '27', '29', '31', '49', '54', '63', '74')
    ALP_THRESHOLDS = ('<0.5m', '<1m', '<2m')
    OUR_METHODS = ['monoloco_pp', 'monstereo']#['geometric', 'monoloco', 'monoloco_pp', 'pose', 'reid', 'monstereo']
    #METHODS_MONO = ['m3d', 'monodis', 'monogrnet', 'smoke']#, 'monopsr']
    METHODS_MONO = ['m3e', 'm4e']#['m3d']#, 'monopsr']
    METHODS_STEREO = ['3dop', 'pseudo-lidar']#['3dop', 'psf', 'pseudo-lidar', 'e2e', 'oc-stereo']
    BASELINES = []#'task_error', 'pixel_error']
    HEADERS = ('method', '<0.5', '<1m', '<2m', 'easy', 'moderate', 'hard', 'all')
    CATEGORIES = ('pedestrian',)

    def __init__(self, thresh_iou_monoloco=0.3, thresh_iou_base=0.3, thresh_conf_monoloco=0.2 , thresh_conf_base=0.5,
                 verbose=False, vehicles = False, transformer = False,dir_ann = None, logger = None):

        self.main_dir = os.path.join('data', 'kitti')

        self.vehicles= vehicles
        self.dir_gt = os.path.join(self.main_dir, 'training', "label_2")
        self.methods = self.OUR_METHODS + self.METHODS_MONO + self.METHODS_STEREO

        self.categories = self.CATEGORIES if not vehicles else ('car', )

        self.identifier=''
        if vehicles:
            self.identifier+='car-'
        else:
            self.identifier+='human-'

        if transformer:
            self.identifier+="transformer-"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M%S")[2:]
        name_out = 'ms-' + now_time+'-'+self.identifier+"eval"+".txt"
        if logger is not None:
            self.logger = logger
        else:
            self.logger = set_logger(os.path.join('data', 'logs', name_out))


        path_train = os.path.join('splits', 'kitti_train.txt')
        path_val = os.path.join('splits', 'kitti_val.txt')
        dir_logs = os.path.join('data', 'logs')
        assert dir_logs, "No directory to save final statistics"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        #self.path_results = os.path.join(dir_logs, 'eval-' + now_time + '.json')
        self.verbose = verbose

        self.dic_thresh_iou = {method: (thresh_iou_monoloco if method in self.OUR_METHODS
                                        else thresh_iou_base)
                               for method in self.methods}
        self.dic_thresh_conf = {method: (thresh_conf_monoloco if method in self.OUR_METHODS
                                         else thresh_conf_base)
                                for method in self.methods}

        if not vehicles:
            try:
                indexes = [self.methods.index('pseudo-lidar'), self.methods.index('monogrnet')]
                for index in indexes:
                    self.methods.pop(index)
            except:
                pass
            print(self.methods)
            self.logger.info(self.methods)
        if 'monopsr' in self.methods:
            self.dic_thresh_conf['monopsr'] += 0.3
        self.dic_thresh_conf['e2e-pl'] = -100  # They don't have enough detections
        self.dic_thresh_conf['oc-stereo'] = -100

        self.logger.info("dataset used for the evaluation: {}".format(dir_ann))
        # Extract validation images for evaluation
        names_gt = tuple(os.listdir(self.dir_gt))
        _, self.set_val = split_training(names_gt, path_train, path_val)

        # Define variables to save statistics
        self.dic_methods = self.errors = self.dic_stds = self.dic_stats = self.dic_cnt = self.cnt_gt = self.category \
            = None
        self.cnt = 0

    def run(self):
        """Evaluate Monoloco performances on ALP and ALE metrics"""
        for self.category in self.categories:#self.CATEGORIES:

            print(self.category)
            # Initialize variables
            self.errors = defaultdict(lambda: defaultdict(list))
            self.dic_stds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            self.dic_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
            self.dic_cnt = defaultdict(int)
            self.cnt_gt = defaultdict(int)

            # Iterate over each ground truth file in the training set
            # self.set_val = ('000063.txt',)
            for name in self.set_val:
                path_gt = os.path.join(self.dir_gt, name)
                self.name = name

                # Iterate over each line of the gt file and save box location and distances
                out_gt = parse_ground_truth(path_gt, self.category)
                methods_out = defaultdict(tuple)  # Save all methods for comparison

                # Count ground_truth:
                boxes_gt, ys, truncs_gt, occs_gt = out_gt
                for idx, box in enumerate(boxes_gt):
                    mode = get_difficulty(box, truncs_gt[idx], occs_gt[idx])
                    self.cnt_gt[mode] += 1
                    self.cnt_gt['all'] += 1

                if out_gt[0]:
                    for method in self.methods:
                        # Extract annotations
                        dir_method = os.path.join(self.main_dir, method)
                        assert os.path.exists(dir_method), "directory of the method %s does not exists" % method
                        path_method = os.path.join(dir_method, name)
                        methods_out[method] = self._parse_txts(path_method, method=method)

                        # Compute the error with ground truth
                        self._estimate_error(out_gt, methods_out[method], method=method)

            # Update statistics of errors and uncertainty
            for key in self.errors:
                add_true_negatives(self.errors[key], self.cnt_gt['all'])
                for clst in self.CLUSTERS[:-1]:

                    try:
                        get_statistics(self.dic_stats['test'][key][clst],
                                       self.errors[key][clst],
                                       self.dic_stds[key][clst], key)
                    except ZeroDivisionError:
                        print('\n'+'-'*100 + '\n'+f'ERROR: method {key} at cluster {clst} is empty' + '\n'+'-'*100+'\n')

            # Show statistics
            print('\n' + self.category.upper() + ':')
            #self.show_statistics()
            self.show_statistics_logger()

    def printer(self, show, save):

        if self.vehicles:
            identifier = "_vehicles"
        else:
            identifier = "_human"

        
        if save or show:
            show_results(self.dic_stats,self.methods ,self.CLUSTERS, show, save, identifier=identifier, vehicles = self.vehicles)
            show_spread(self.dic_stats, self.CLUSTERS, show, save, identifier=identifier)
            show_box_plot(self.errors, self.CLUSTERS, show, save, identifier=identifier)
            show_task_error(show, save, identifier=identifier)

    def _parse_txts(self, path, method):

        boxes = []
        dds = []
        cat = []

        if method == 'psf':
            path = os.path.splitext(path)[0] + '.png.txt'
        if method in self.OUR_METHODS:
            bis, epis = [], []
            output = (boxes, dds, cat, bis, epis)
        else:
            output = (boxes, dds, cat)
        try:
            with open(path, "r") as ff:
                for line_str in ff:
                    if method == 'psf':
                        line = line_str.split(", ")
                        box = [float(x) for x in line[4:8]]
                        boxes.append(box)
                        loc = ([float(x) for x in line[11:14]])
                        dd = math.sqrt(loc[0] ** 2 + loc[1] ** 2 + loc[2] ** 2)
                        dds.append(dd)
                        cat.append('Pedestrian'if not self.vehicles else 'Car')
                    else:
                        line = line_str.split()
                        if check_conditions(line,
                                            category='pedestrian' if not self.vehicles else 'car',
                                            method=method,
                                            thresh=self.dic_thresh_conf[method]):
                            box = [float(x) for x in line[4:8]]
                            box.append(float(line[15]))  # Add confidence
                            loc = ([float(x) for x in line[11:14]])
                            dd = math.sqrt(loc[0] ** 2 + loc[1] ** 2 + loc[2] ** 2)
                            cat.append(line[0])
                            boxes.append(box)
                            dds.append(dd)
                            if method in self.OUR_METHODS:
                                bis.append(float(line[16]))
                                epis.append(float(line[17]))
                            self.dic_cnt[method] += 1

            return output
        except FileNotFoundError:
            return output

    def _estimate_error(self, out_gt, out, method):
        """Estimate localization error"""

        boxes_gt, ys, truncs_gt, occs_gt = out_gt

        if method in self.OUR_METHODS:
            boxes, dds, cat, bis, epis = out
        else:
            boxes, dds, cat = out

        if method == 'psf':
            matches = get_iou_matches_matrix(boxes, boxes_gt, self.dic_thresh_iou[method])
        else:
            matches = get_iou_matches(boxes, boxes_gt, self.dic_thresh_iou[method])

        for (idx, idx_gt) in matches:
            # Update error if match is found
            dd_gt = ys[idx_gt][3]
            zz_gt = ys[idx_gt][2]
            mode = get_difficulty(boxes_gt[idx_gt], truncs_gt[idx_gt], occs_gt[idx_gt])

            if cat[idx].lower() in (self.category, 'pedestrian' if not self.vehicles else 'car'):
                self.update_errors(dds[idx], dd_gt, mode, self.errors[method])
                if method == 'monoloco':
                    dd_task_error = dd_gt + (get_task_error(zz_gt))**2
                    dd_pixel_error = dd_gt + get_pixel_error(zz_gt)
                    self.update_errors(dd_task_error, dd_gt, mode, self.errors['task_error'])
                    self.update_errors(dd_pixel_error, dd_gt, mode, self.errors['pixel_error'])
                if method in self.OUR_METHODS:
                    epi = max(epis[idx], bis[idx])
                    self.update_uncertainty(bis[idx], epi, dds[idx], dd_gt, mode, self.dic_stds[method])

    def update_errors(self, dd, dd_gt, cat, errors):
        """Compute and save errors between a single box and the gt box which match"""
        diff = abs(dd - dd_gt)
        clst = find_cluster(dd_gt, self.CLUSTERS[4:])
        errors['all'].append(diff)
        errors[cat].append(diff)
        errors[clst].append(diff)

        # Check if the distance is less than one or 2 meters
        if diff <= 0.5:
            errors['<0.5m'].append(1)
        else:
            errors['<0.5m'].append(0)

        if diff <= 1:
            errors['<1m'].append(1)
        else:
            errors['<1m'].append(0)

        if diff <= 2:
            errors['<2m'].append(1)
        else:
            errors['<2m'].append(0)

    def update_uncertainty(self, std_ale, std_epi, dd, dd_gt, mode, dic_stds):

        clst = find_cluster(dd_gt, self.CLUSTERS[4:])
        dic_stds['all']['ale'].append(std_ale)
        dic_stds[clst]['ale'].append(std_ale)
        dic_stds[mode]['ale'].append(std_ale)
        dic_stds['all']['epi'].append(std_epi)
        dic_stds[clst]['epi'].append(std_epi)
        dic_stds[mode]['epi'].append(std_epi)
        dic_stds['all']['epi_rel'].append(std_epi / dd)
        dic_stds[clst]['epi_rel'].append(std_epi / dd)
        dic_stds[mode]['epi_rel'].append(std_epi / dd)

        # Number of annotations inside the confidence interval
        std = std_epi if std_epi > 0 else std_ale  # consider aleatoric uncertainty if epistemic is not calculated
        if abs(dd - dd_gt) <= std:
            dic_stds['all']['interval'].append(1)
            dic_stds[clst]['interval'].append(1)
            dic_stds[mode]['interval'].append(1)
        else:
            dic_stds['all']['interval'].append(0)
            dic_stds[clst]['interval'].append(0)
            dic_stds[mode]['interval'].append(0)

        # Annotations at risk inside the confidence interval
        if dd_gt <= dd:
            dic_stds['all']['at_risk'].append(1)
            dic_stds[clst]['at_risk'].append(1)
            dic_stds[mode]['at_risk'].append(1)

            if abs(dd - dd_gt) <= std_epi:
                dic_stds['all']['at_risk-interval'].append(1)
                dic_stds[clst]['at_risk-interval'].append(1)
                dic_stds[mode]['at_risk-interval'].append(1)
            else:
                dic_stds['all']['at_risk-interval'].append(0)
                dic_stds[clst]['at_risk-interval'].append(0)
                dic_stds[mode]['at_risk-interval'].append(0)

        else:
            dic_stds['all']['at_risk'].append(0)
            dic_stds[clst]['at_risk'].append(0)
            dic_stds[mode]['at_risk'].append(0)

        # Precision of uncertainty
        eps = 1e-4
        task_error = get_task_error(dd)
        prec_1 = abs(dd - dd_gt) / (std_epi + eps)

        prec_2 = abs(std_epi - task_error)
        dic_stds['all']['prec_1'].append(prec_1)
        dic_stds[clst]['prec_1'].append(prec_1)
        dic_stds[mode]['prec_1'].append(prec_1)
        dic_stds['all']['prec_2'].append(prec_2)
        dic_stds[clst]['prec_2'].append(prec_2)
        dic_stds[mode]['prec_2'].append(prec_2)


    def show_statistics_logger(self):

        all_methods = self.methods + self.BASELINES
        self.logger.info('-'*90)
        self.summary_table_logger(all_methods)

        # Uncertainty
        for net in ('monoloco_pp', 'monstereo'):
            self.logger.info(('-'*100))
            self.logger.info(net.upper())

            try:
                process_mode = os.environ["process_mode"]
            except:
                process_mode = "NULL"

            self.logger.info("Process mode: {}".format(process_mode))
            try:
                dropout_images = os.environ["dropout"]
            except:
                dropout_images = "NULL"

            self.logger.info("Dropout images: {}".format(dropout_images))

            
            for clst in ('easy', 'moderate', 'hard', 'all'):
                self.logger.info(" Annotations in clst {}: {:.0f}, Recall: {:.1f}. Precision: {:.2f}, Relative size is {:.1f} %"
                      .format(clst,
                              self.dic_stats['test'][net][clst]['cnt'],
                              self.dic_stats['test'][net][clst]['interval']*100,
                              self.dic_stats['test'][net][clst]['prec_1'],
                              self.dic_stats['test'][net][clst]['epi_rel']*100))

        if self.verbose:
            for key in all_methods:
                
                for clst in self.CLUSTERS[:4]:
                    self.logger.info(" {} Average error in cluster {}: {:.2f} with a max error of {:.1f}, "
                          "for {} annotations"
                          .format(key, clst, self.dic_stats['test'][key][clst]['mean'],
                                  self.dic_stats['test'][key][clst]['max'],
                                  self.dic_stats['test'][key][clst]['cnt']))

                for perc in self.ALP_THRESHOLDS:
                    self.logger.info("{} Instances with error {}: {:.2f} %"
                          .format(key, perc, 100 * average(self.errors[key][perc])))
                try:
                    self.logger.info("\nMatched annotations: {:.1f} %".format(self.errors[key]['matched']))
                    self.logger.info(" Detected annotations : {}/{} ".format(self.dic_cnt[key], self.cnt_gt['all']))
                    self.logger.info("-" * 100)
                except TypeError:
                    self.logger.info("Nothing detected")
            self.logger.info("precision 1: {:.2f}".format(self.dic_stats['test']['monoloco']['all']['prec_1']))
            self.logger.info("precision 2: {:.2f}".format(self.dic_stats['test']['monoloco']['all']['prec_2']))


    def summary_table_logger(self, all_methods):
        """Tabulate table for ALP and ALE metrics"""

        self.logger.info("METHODS")

        alp = [[str(100 * average(self.errors[key][perc]))[:5]
                for perc in ['<0.5m', '<1m', '<2m']]
               for key in all_methods]

        ale = [[str(round(self.dic_stats['test'][key][clst]['mean'], 3))[:4] + ' [' +
                str(round(self.dic_stats['test'][key][clst]['cnt'] / self.cnt_gt[clst] * 100))[:4] + '%]'
                for clst in self.CLUSTERS[:4]]
               for key in all_methods]

        results = [[key] + alp[idx] + ale[idx] for idx, key in enumerate(all_methods)]
        self.logger.info("\n"+ tabulate(results, headers=self.HEADERS))
        self.logger.info('-' * 90 + '\n')


    def show_statistics(self):

        all_methods = self.methods + self.BASELINES
        print('-'*90)
        self.summary_table(all_methods)

        # Uncertainty
        for net in ('monoloco_pp', 'monstereo'):
            print(('-'*100))
            print(net.upper())
            for clst in ('easy', 'moderate', 'hard', 'all'):
                print(" Annotations in clst {}: {:.0f}, Recall: {:.1f}. Precision: {:.2f}, Relative size is {:.1f} %"
                      .format(clst,
                              self.dic_stats['test'][net][clst]['cnt'],
                              self.dic_stats['test'][net][clst]['interval']*100,
                              self.dic_stats['test'][net][clst]['prec_1'],
                              self.dic_stats['test'][net][clst]['epi_rel']*100))

        if self.verbose:
            for key in all_methods:
                print(key.upper())
                for clst in self.CLUSTERS[:4]:
                    print(" {} Average error in cluster {}: {:.2f} with a max error of {:.1f}, "
                          "for {} annotations"
                          .format(key, clst, self.dic_stats['test'][key][clst]['mean'],
                                  self.dic_stats['test'][key][clst]['max'],
                                  self.dic_stats['test'][key][clst]['cnt']))

                for perc in self.ALP_THRESHOLDS:
                    print("{} Instances with error {}: {:.2f} %"
                          .format(key, perc, 100 * average(self.errors[key][perc])))
                try:
                    print("\nMatched annotations: {:.1f} %".format(self.errors[key]['matched']))
                    print(" Detected annotations : {}/{} ".format(self.dic_cnt[key], self.cnt_gt['all']))
                    print("-" * 100)
                except TypeError:
                    print("Nothing detected")
            print("precision 1: {:.2f}".format(self.dic_stats['test']['monoloco']['all']['prec_1']))
            print("precision 2: {:.2f}".format(self.dic_stats['test']['monoloco']['all']['prec_2']))

    def summary_table(self, all_methods):
        """Tabulate table for ALP and ALE metrics"""

        print("METHODS", all_methods)

        alp = [[str(100 * average(self.errors[key][perc]))[:5]
                for perc in ['<0.5m', '<1m', '<2m']]
               for key in all_methods]

        ale = [[str(round(self.dic_stats['test'][key][clst]['mean'], 3))[:] + ' [' +
                str(round(self.dic_stats['test'][key][clst]['cnt'] / self.cnt_gt[clst] * 100))[:3] + '%]'
                for clst in self.CLUSTERS[:4]]
               for key in all_methods]

        ale = [[str(self.dic_stats['test'][key][clst]['mean'])[:] + ' [' +
                str(round(self.dic_stats['test'][key][clst]['cnt'] / self.cnt_gt[clst] * 100))[:3] + '%]'
                for clst in self.CLUSTERS[:4]]
               for key in all_methods]

        print(ale)
        results = [[key] + alp[idx] + ale[idx] for idx, key in enumerate(all_methods)]
        print(tabulate(results, headers=self.HEADERS))
        print('-' * 90 + '\n')

    def stats_height(self):
        heights = []
        for name in self.set_val:
            path_gt = os.path.join(self.dir_gt, name)
            self.name = name
            # Iterate over each line of the gt file and save box location and distances
            out_gt = parse_ground_truth(path_gt, 'pedestrian')
            boxes_gt, ys, truncs_gt, occs_gt = out_gt
            for label in ys:
                heights.append(label[4])
        tail1, tail2 = np.nanpercentile(np.array(heights), [5, 95])
        print(average(heights))
        print(len(heights))
        print(tail1, tail2)


def get_statistics(dic_stats, errors, dic_stds, key):
    """Update statistics of a cluster"""

    try:
        dic_stats['mean'] = average(errors)
        dic_stats['max'] = max(errors)
        dic_stats['cnt'] = len(errors)
    except ValueError:
        dic_stats['mean'] = - 1
        dic_stats['max'] = - 1
        dic_stats['cnt'] = - 1

    if key in ('monoloco', 'monoloco_pp', 'monstereo'):
        dic_stats['std_ale'] = average(dic_stds['ale'])
        dic_stats['std_epi'] = average(dic_stds['epi'])
        dic_stats['epi_rel'] = average(dic_stds['epi_rel'])
        dic_stats['interval'] = average(dic_stds['interval'])
        dic_stats['at_risk'] = average(dic_stds['at_risk'])
        dic_stats['prec_1'] = average(dic_stds['prec_1'])
        dic_stats['prec_2'] = average(dic_stds['prec_2'])


def add_true_negatives(err, cnt_gt):
    """Update errors statistics of a specific method with missing detections"""

    matched = len(err['all'])
    missed = cnt_gt - matched
    zeros = [0] * missed
    err['<0.5m'].extend(zeros)
    err['<1m'].extend(zeros)
    err['<2m'].extend(zeros)
    err['matched'] = 100 * matched / cnt_gt


def find_cluster(dd, clusters): #! Outdated function. Does not work with the current value inputed. PLS do not consider.
    """Find the correct cluster. Above the last cluster goes into "excluded (together with the ones from kitti cat"""

    for idx, clst in enumerate(clusters[:-1]):  #easy, moderate, hard
        if int(clst) < dd <= int(clusters[idx+1]):
            return clst
    return 'excluded'


def extract_indices(idx_to_check, *args):
    """
    Look if a given index j_gt is present in all the other series of indices (_, j)
    and return the corresponding one for argument

    idx_check --> gt index to check for correspondences in other method
    idx_method --> index corresponding to the method
    idx_gt --> index gt of the method
    idx_pred --> index of the predicted box of the method
    indices --> list of predicted indices for each method corresponding to the ground truth index to check
    """

    checks = [False]*len(args)
    indices = []
    for idx_method, method in enumerate(args):
        for (idx_pred, idx_gt) in method:
            if idx_gt == idx_to_check:
                checks[idx_method] = True
                indices.append(idx_pred)
    return all(checks), indices


def average(my_list):
    """calculate mean of a list"""
    if len(my_list) == 0:
        return -1

    return sum(my_list) / len(my_list)
