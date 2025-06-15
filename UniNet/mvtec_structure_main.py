import numpy as np
import os
from tabulate import tabulate
from train_unsupervisedAD import train
from datasets import unsupervised, industrial, mvtec_list
import argparse
from utils import setup_seed, get_logger
from test import test


def parsing_args():
    parser = argparse.ArgumentParser(description='UniNet')

    parser.add_argument('--domain', default='industrial', type=str,
                        choices=['industrial', 'medical', 'video', 'natural'], help="choose experimental domain.")
    parser.add_argument('--setting', default='mc', type=str, choices=['oc', 'mc', 'cd'],
                        help="choose experimental settings, including one-class, multi-class, cross-dataset.")
    parser.add_argument('--dataset', default='VisA', type=str,
                        choices=['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA", "VAD", "APTOS", "ISIC2018", "OCT2017",
                                 "Kvasir-SEG", "CVC-ClinicDB", "CVC-ColonDB", "Ped2"],
                        help="choose experimental dataset.")
    parser.add_argument('--epochs', default=100, type=int, help="epochs.")
    parser.add_argument('--batch_size', default=8, type=int, help="batch sizes.")
    parser.add_argument('--image_size', default=256, type=int, help="image size.")
    parser.add_argument('--center_crop', default=256, type=int, help="crop image size.")
    parser.add_argument('--lr_s', default=5e-3, type=float, help="lr for student.")  # 5e-3
    parser.add_argument('--lr_t', default=1e-6, type=float, help="lr for teacher.")  # 1e-6
    parser.add_argument('--T', default=2, type=float, help="temperature for contrastive learning.")

    parser.add_argument('--weighted_decision_mechanism', action='store_true', default=True,
                        help='whether to employ weight-guided similarity to calculate anomaly map.')
    parser.add_argument('--default', default=0.3, type=float, help='the default value of weights.')
    parser.add_argument('--alpha', default=0.01, type=float, help='hyperparameters for weights.')
    parser.add_argument('--beta', default=0.00003, type=float, help='hyperparameters for weights.')

    parser.add_argument("--train_and_test_all", action='store_true', default=False,
                        help="for medical domains.")
    parser.add_argument("--is_saved", action='store_true', default=True, help="whether to save model weights.")
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--load_ckpts', action='store_true', default=False, help="loading ckpts for testing")

    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(1203)
    args = parsing_args()
    if not args.weighted_decision_mechanism:
        args.default = args.alpha = args.beta = args.gamma = "w/o"

    args.domain = 'industrial'
    dataset_name = args.dataset
    logger = get_logger(dataset_name, os.path.join(args.save_dir, dataset_name))
    dataset = mvtec_list

    results = {}
    result_tabel_rows = []

    image_auroc_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []

    # -----------------------------train-------------------------------------
    if not args.load_ckpts:
        for idx, i in enumerate(dataset):
            args._class_ = i
            args_dict = vars(args)
            args_info = f"class:{i}, " if args.setting == 'oc' else f""
            for key, value in args_dict.items():
                if key in ['_class_']:
                    continue
                args_info += ", ".join([f"{key}:{value}, "])

            if args.setting == 'oc':
                print('training on {} dataset (separate-class)'.format(dataset_name)) if idx == 0 else None
                print(args_info)
                train(args)

            elif args.setting == 'mc':
                if dataset_name == 'MVTec AD':
                    args.T = 0.1
                print('training on {} dataset (multiclass)'.format(dataset_name)) if idx == 0 else None
                print(args_info)
                from train_multiclass import train_mc
                train_mc(args)
                break

            else:
                pass

            print()
        print("training over!")

    # -----------------------------test-------------------------------------
    if args.setting == 'oc':
        for idx, i in enumerate(dataset):
            print('testing on {} dataset (separate-class)'.format(dataset_name)) if idx == 0 else None
            args._class_ = i
            print(f"testing class:{i}")
            auroc_sp, auroc_px, aupro_px = test(args, suffix='BEST_P_PRO')
            print('')
            result_tabel_rows.append(['{}'.format(i), str(np.round(auroc_sp, decimals=1)),
                                      str(np.round(auroc_px, decimals=1)),
                                      str(np.round(aupro_px, decimals=1))])
            image_auroc_list.append(auroc_sp)
            pixel_auroc_list.append(auroc_px)
            pixel_aupro_list.append(aupro_px)
            results = tabulate(result_tabel_rows, headers=['object', 'image_auroc', 'pixel_auroc', 'pixel_aupro'],
                               tablefmt="pipe")
        result_tabel_rows.append(['mean', str(np.round(np.mean(image_auroc_list), decimals=2)),
                                  str(np.round(np.mean(pixel_auroc_list), decimals=2)),
                                  str(np.round(np.mean(pixel_aupro_list), decimals=2))])
        results = tabulate(result_tabel_rows, headers=['object', 'image_auroc', 'pixel_auroc', 'pixel_aupro'],
                           tablefmt="pipe")
        print(results)

    elif args.setting == 'mc':
        print('testing on {} dataset (multiclass)'.format(dataset_name))

        args._class_ = ''
        auroc_list, acc_list, f1_list, class_list = test(args, suffix='BEST_I_ROC')
        print('')

        for i, x, y, z in zip(class_list, auroc_list, acc_list, f1_list):
            result_tabel_rows.append(['{}'.format(i), str(np.round(x, decimals=1)),
                                      str(np.round(y, decimals=1)),
                                      str(np.round(z, decimals=1))])
            results = tabulate(result_tabel_rows, headers=['object', 'image_auroc', 'acc', 'f1'], tablefmt="pipe")
        result_tabel_rows.append(['mean', str(np.round(np.mean(auroc_list), decimals=2)),
                                  str(np.round(np.mean(acc_list), decimals=2)),
                                  str(np.round(np.mean(f1_list), decimals=2))])
        results = tabulate(result_tabel_rows, headers=['object', 'image_auroc', 'acc', 'f1'],
                           tablefmt="pipe")
        print(results)
