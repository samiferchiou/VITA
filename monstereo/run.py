# pylint: disable=too-many-branches, too-many-statements

import argparse

#from openpifpaf.network import nets
#from openpifpaf import decoder

def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Subparser definition
    subparsers = parser.add_subparsers(help='Different parsers for main actions', dest='command')
    predict_parser = subparsers.add_parser("predict")
    prep_parser = subparsers.add_parser("prep")
    training_parser = subparsers.add_parser("train")
    eval_parser = subparsers.add_parser("eval")

    # Preprocess input data
    prep_parser.add_argument('--dir_ann', help='directory of annotations of 2d joints', required=True)
    prep_parser.add_argument('--dataset',
                             help="datasets to preprocess: nuscenes, nuscenes_teaser, nuscenes_mini, kitti, apolloscape ,apolloscape-mini",
                             default='kitti')
    prep_parser.add_argument('--dir_nuscenes', help='directory of nuscenes devkit', default='data/nuscenes/')
    prep_parser.add_argument('--dir_apolloscape', help='directory of the apolloscape dataser', default='data/apolloscape/' )
    prep_parser.add_argument('--kps_3d', help='Change the output size of the network to train the network on the 3D position of the keypoints [Warning, '
                            'this is only available for the apolloscape vehicle datset with 24 keypoints]', action='store_true')
    prep_parser.add_argument('--pifpaf_kps', help='indicates that we are using the keypoints processed ', action='store_true')
    prep_parser.add_argument('--iou_min', help='minimum iou to match ground truth', type=float, default=0.3)
    prep_parser.add_argument('--variance', help='new', action='store_true')
    prep_parser.add_argument('--activity', help='new', action='store_true')
    prep_parser.add_argument('--monocular', help='new', action='store_true')
    prep_parser.add_argument('--vehicles', help="Indicate that we are training,evaluating or predicting vehicles position instead of human's one", action ='store_true')
    prep_parser.add_argument('--buffer', help='indicates the quantity of keypoints used from the car models to do the assignment between 2D adn 3D keypoints', default = 20)
    prep_parser.add_argument('--radius', help='Radius to determine wether one set of keypoint can be assimilated to which vehicle in the re-projected 3D model view', default = 200)
    prep_parser.add_argument('--dropout', help='Extend the dataset by providing a dropout on some of its keypoints with a probability of : dropout', type=float, default = 0.0)
    prep_parser.add_argument('--confidence', help='Add the confidences of the keypoints in the processing loop ', action='store_true')
    prep_parser.add_argument('--transformer', help='Use a Transformer as the encoder of the Neural network', action = 'store_true')
    prep_parser.add_argument('--lstm', help='Use an LSTM for the processing', action = 'store_true')
    prep_parser.add_argument('--scene_disp', help='Use a batchification by scenes', action = 'store_true')
    prep_parser.add_argument('--scene_refine', help='Use a refining step after the use of the transformer', action = 'store_true')




    # Predict (2D pose and/or 3D location from images)
    # General
    predict_parser.add_argument('--mode', help='pifpaf, mono, stereo', default='stereo')
    predict_parser.add_argument('images', nargs='*', help='input images')
    predict_parser.add_argument('--glob', help='glob expression for input images (for many images)')
    predict_parser.add_argument('-o', '--output-directory', help='Output directory', default='visual_tests/')
    predict_parser.add_argument('--output_types', nargs='+', default=['json'],
                                help='what to output: json keypoints skeleton for Pifpaf'
                                     'json bird front combined for Monoloco')
    predict_parser.add_argument('--show', help='to show images', action='store_true')
    predict_parser.add_argument('--joints_folder', help='Folder containing the pifpaf annotations',default=None)
    predict_parser.add_argument('--vehicles', help="Indicate that we are training,evaluating or predicting vehicles position instead of human's one", action ='store_true')
    predict_parser.add_argument('--kps_3d', help='Change the output size of the network to train the network on the 3D position of the keypoints [Warning, '
                            'this is only available for the apolloscape vehicle datset with 24 keypoints]', action='store_true')
    predict_parser.add_argument('--confidence', help='Add the confidences of the keypoints in the processing loop ', action='store_true')
    predict_parser.add_argument('--transformer', help='Use a Transformer as the encoder of the Neural network', action = 'store_true')
    predict_parser.add_argument('--scene_refine', help='Use a refining step after the use of the transformer', action = 'store_true')
    predict_parser.add_argument('--lstm', help='Use an LSTM for the processing', action = 'store_true')
    predict_parser.add_argument('--scene_disp', help='Uses the scene formatting', action = 'store_true')


    # Pifpaf
    predict_parser.add_argument('--scale', default=1.0, type=float, help='change the scale of the image to preprocess')
    predict_parser.add_argument('--checkpoint', help='PifPaf model to load', default='shufflenetv2k16-apollo-24')


    # Monoloco
    predict_parser.add_argument('--model', help='path of MonoLoco model to load', required=True)
    predict_parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=512)
    predict_parser.add_argument('--path_gt', help='path of json file with gt 3d localization',
                                default='data/arrays/names-kitti-200615-1022.json')

    #
    predict_parser.add_argument('--transform', help='transformation for the pose', default='None')
    predict_parser.add_argument('--draw_box', help='to draw box in the images', action='store_true')
    predict_parser.add_argument('--z_max', type=int, help='maximum meters distance for predictions', default=22)
    predict_parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    predict_parser.add_argument('--dropout', type=float, help='dropout parameter', default=0.2)
    predict_parser.add_argument('--show_all', help='only predict ground-truth matches or all', action='store_true')

    # Social distancing and social interactions
    predict_parser.add_argument('--social', help='social', action='store_true')
    predict_parser.add_argument('--activity', help='activity', action='store_true')
    predict_parser.add_argument('--json_dir', help='for social')
    predict_parser.add_argument('--threshold_prob', type=float, help='concordance for samples', default=0.25)
    predict_parser.add_argument('--threshold_dist', type=float, help='min distance of people', default=2)
    predict_parser.add_argument('--margin', type=float, help='conservative for noise in orientation', default=1.5)
    predict_parser.add_argument('--radii', type=tuple, help='o-space radii', default=(0.25, 1, 2))

    # Training
    training_parser.add_argument('--joints', help='Json file with input joints',
                                 default='data/arrays/joints-nuscenes_teaser-190513-1846.json')
    training_parser.add_argument('--save', help='whether to not save model and log file', action='store_true')
    training_parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train for', default=500)
    training_parser.add_argument('--bs', type=int, default=512, help='input batch size')
    training_parser.add_argument('--monocular', help='whether to train monoloco', action='store_true')#TRUE for us
    training_parser.add_argument('--dataset', help='datasets to evaluate, kitti, nuscenes or apolloscape', default='kitti')
    training_parser.add_argument('--kps_3d', help='Change the output size of the network to train the network on the 3D position of the keypoints [Warning, '
                            'this is only available for the apolloscape vehicle datset with 24 keypoints]', action='store_true')
    training_parser.add_argument('--dropout', type=float, help='dropout. Default no dropout', default=0.2)
    training_parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    training_parser.add_argument('--sched_step', type=float, help='scheduler step time (epochs)', default=30)
    training_parser.add_argument('--sched_gamma', type=float, help='Scheduler multiplication every step', default=0.98)
    training_parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=1024)
    training_parser.add_argument('--n_stage', type=int, help='Number of stages in the model', default=3)
    training_parser.add_argument('--hyp', help='run hyperparameters tuning', action='store_true')
    training_parser.add_argument('--multiplier', type=int, help='Size of the grid of hyp search', default=1)
    training_parser.add_argument('--r_seed', type=int, help='specify the seed for training and hyp tuning', default=1)
    training_parser.add_argument('--activity', help='new', action='store_true')
    training_parser.add_argument('--vehicles', help="Indicate that we are training,evaluating or predicting vehicles position instead of human's one", action ='store_true')
    training_parser.add_argument('--confidence', help='Add the confidences of the keypoints in the processing loop ', action='store_true')
    training_parser.add_argument('--transformer', help='Use a Transformer as the encoder of the Neural network', action = 'store_true')
    training_parser.add_argument('--lstm', help='Use an LSTM for the processing', action = 'store_true')
    training_parser.add_argument('--scene_disp', help='Use a batchification by scenes', action = 'store_true')
    training_parser.add_argument('--scene_refine', help='Use a refining step after the use of the transformer', action = 'store_true')
    training_parser.add_argument('--dir_ann', help='directory of annotations of 2d joints (for KITTI evaluation)')
    training_parser.add_argument('--num_heads', type=int, help='Number of heads for the multi-headed attention mechanism', default = 4)


    # Evaluation
    eval_parser.add_argument('--dataset', help='datasets to evaluate, kitti, nuscenes or apolloscape', default='kitti')
    eval_parser.add_argument('--geometric', help='to evaluate geometric distance', action='store_true')
    eval_parser.add_argument('--generate', help='create txt files for KITTI evaluation', action='store_true')
    eval_parser.add_argument('--dir_ann', help='directory of annotations of 2d joints (for KITTI evaluation)')
    eval_parser.add_argument('--model', help='path of MonoLoco model to load', default=None)
    eval_parser.add_argument('--joints', help='Json file with input joints to evaluate (for nuScenes evaluation)')
    eval_parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    eval_parser.add_argument('--dropout', type=float, help='dropout. Default no dropout', default=0.2)
    eval_parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=1024)
    eval_parser.add_argument('--n_stage', type=int, help='Number of stages in the model', default=3)
    eval_parser.add_argument('--show', help='whether to show statistic graphs', action='store_true')
    eval_parser.add_argument('--save', help='whether to save statistic graphs', action='store_true')
    eval_parser.add_argument('--verbose', help='verbosity of statistics', action='store_true')
    eval_parser.add_argument('--monocular', help='whether to train using the baseline', action='store_true')
    eval_parser.add_argument('--new', help='new', action='store_true')
    eval_parser.add_argument('--variance', help='evaluate keypoints variance', action='store_true')
    eval_parser.add_argument('--activity', help='evaluate activities', action='store_true')
    eval_parser.add_argument('--net', help='Choose network: monoloco, monoloco_p, monoloco_pp, monstereo')
    eval_parser.add_argument('--kps_3d', help='Change the output size of the network to train the network on the 3D position of the keypoints [Warning, '
                            'this is only available for the apolloscape vehicle datset with 24 keypoints]', action='store_true')
    eval_parser.add_argument('--vehicles', help="Indicate that we are training,evaluating or predicting vehicles position instead of human's one", action ='store_true')
    eval_parser.add_argument('--model_mono', help='mono model that can be added to compute the score evaluation for monoloco_pp', default = None)
    eval_parser.add_argument('--confidence', help='Add the confidences of the keypoints in the processing loop ', action='store_true')
    eval_parser.add_argument('--transformer', help='Use a Transformer as the encoder of the Neural network', action = 'store_true')
    eval_parser.add_argument('--lstm', help='Use an LSTM for the processing', action = 'store_true')
    eval_parser.add_argument('--scene_disp', help='Use a batchification by scenes', action = 'store_true')
    eval_parser.add_argument('--scene_refine', help='Use a refining step after the use of the transformer', action = 'store_true')
    args = parser.parse_args()
    return args


def main():
    args = cli()

    if args.transformer and args.command == 'prep':
        #? For the preparation step with the transformer, we want to have the confidence as well
        #? as the 2d coordinate of the keypoints
        #? The confidence is always needed since it is used to obtain the mask for the inputs.
        args.confidence=True
    if args.command == 'predict':
        if args.activity:
            from .activity import predict
        else:
            from .predict import predict
        predict(args)

    elif args.command == 'prep':

        if 'nuscenes' in args.dataset:
            from .prep.preprocess_nu import PreprocessNuscenes
            prep = PreprocessNuscenes(args.dir_ann, args.dir_nuscenes, args.dataset, args.iou_min)
            prep.run()
        elif 'apolloscape_mini' in args.dataset:
            from .prep.prep_apolloscape import PreprocessApolloscape
            prep = PreprocessApolloscape(args.dir_ann, dataset = '3d_car_instance_sample', buffer = args.buffer,
                                        kps_3d = args.kps_3d, dropout = args.dropout, confidence = args.confidence,
                                        iou_min = args.iou_min, transformer =args.transformer)
            prep.run()
        elif 'apolloscape' in args.dataset:
            from .prep.prep_apolloscape import PreprocessApolloscape
            prep = PreprocessApolloscape(args.dir_ann, dataset = 'train', buffer = args.buffer,  kps_3d = args.kps_3d,
                                        dropout = args.dropout, confidence = args.confidence, iou_min = args.iou_min,
                                        transformer =args.transformer)
            prep.run()

        else:
            from .prep.prep_kitti import PreprocessKitti
            prep = PreprocessKitti(args.dir_ann, args.iou_min, args.monocular, vehicles= args.vehicles,
                                    dropout = args.dropout, confidence=args.confidence, transformer = args.transformer)
            if args.activity:
                prep.prep_activity()
            else:
                prep.run()

    elif args.command == 'train':
        from .train import HypTuning
        if args.hyp:
            hyp_tuning = HypTuning(joints=args.joints, epochs=args.epochs,
                                   monocular=args.monocular, dropout=args.dropout,
                                   multiplier=args.multiplier, r_seed=args.r_seed, 
                                   vehicles = args.vehicles, kps_3d = args.kps_3d,
                                   dataset = args.dataset, confidence = args.confidence,
                                   transformer = args.transformer,
                                   lstm = args.lstm, scene_disp = args.scene_disp,
                                   scene_refine = args.scene_refine,
                                   dir_ann = args.dir_ann)
            hyp_tuning.train()
        else:

            from .train import Trainer
            training = Trainer(joints=args.joints, epochs=args.epochs, bs=args.bs,
                               monocular=args.monocular, dropout=args.dropout, lr=args.lr, sched_step=args.sched_step,
                               n_stage=args.n_stage,  num_heads = args.num_heads, sched_gamma=args.sched_gamma,
                               hidden_size=args.hidden_size, r_seed=args.r_seed, save=args.save, vehicles = args.vehicles,
                               kps_3d = args.kps_3d, dataset = args.dataset, confidence = args.confidence,
                               transformer = args.transformer, lstm = args.lstm, scene_disp= args.scene_disp,
                               scene_refine = args.scene_refine)

            _ = training.train()
            _ = training.evaluate()

    elif args.command == 'eval':

        if args.activity:
            from .eval.eval_activity import ActivityEvaluator
            evaluator = ActivityEvaluator(args)
            if 'collective' in args.dataset:
                evaluator.eval_collective()
            else:
                evaluator.eval_kitti()

        elif args.geometric:
            assert args.joints, "joints argument not provided"
            from .network.geom_baseline import geometric_baseline
            geometric_baseline(args.joints)

        elif args.variance:
            from .eval.eval_variance import joints_variance
            joints_variance(args.joints, clusters=None, dic_ms=None)

        else:
            if args.generate:
                from .eval.generate_kitti import GenerateKitti
                kitti_txt = GenerateKitti(args.model, args.dir_ann, p_dropout=args.dropout, n_dropout=args.n_dropout,
                                        hidden_size=args.hidden_size, vehicles = args.vehicles,
                                        model_mono = args.model_mono, confidence = args.confidence,
                                        transformer = args.transformer, lstm = args.lstm,
                                        scene_disp = args.scene_disp, scene_refine = args.scene_refine)
                kitti_txt.run()

            if args.dataset == 'kitti':
                from .eval import EvalKitti
                kitti_eval = EvalKitti(verbose=args.verbose, vehicles = args.vehicles,
                                        dir_ann=args.dir_ann, transformer=args.transformer)
                kitti_eval.run()
                kitti_eval.printer(show=args.show, save=args.save)

            elif 'nuscenes' in args.dataset:
                from .train import Trainer
                training = Trainer(joints=args.joints, hidden_size=args.hidden_size)
                _ = training.evaluate(load=True, model=args.model, debug=False)

            elif 'apolloscape' in args.dataset:
                from .train import Trainer
                training = Trainer(joints=args.joints, hidden_size=args.hidden_size, dataset=args.dataset,
                                    monocular = args.monocular, vehicles = args.vehicles ,kps_3d = args.kps_3d,
                                    confidence = args.confidence, transformer=args.transformer)
                _ = training.evaluate(load=True, model=args.model, debug=False, confidence = args.confidence,
                                    transformer = args.transformer)
            else:
                raise ValueError("Option not recognized")

    else:
        raise ValueError("Main subparser not recognized or not provided")

if __name__ == '__main__':
    main()
