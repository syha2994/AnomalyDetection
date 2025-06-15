import os
import copy  # 모델 구조를 그대로 복사할 때 사용
import torch
import numpy as np
import wandb
from eval import evaluation_indusAD, evaluation_mediAD, evaluation_video
from utils import save_weights, to_device
from datasets import loading_dataset
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.model import UniNet, EarlyStopping
from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dataset_name = args.dataset
    if args._class_ in [dataset_name]:
        ckpt_path = os.path.join("./ckpts", dataset_name)
    else:
        ckpt_path = os.path.join("./ckpts", dataset_name, f"{args._class_}")

    # ---------------------------------loading dataset-----------------------------------------------
    train_dataloader, test_dataloader = loading_dataset(args, dataset_name)

    # ---------------------------------loading model-------------------------------------------------
    source_teacher, bottleneck = wide_resnet50_2(args, pretrained=True)
    source_teacher.layer4 = None
    source_teacher.fc = None

    target_teacher = copy.deepcopy(source_teacher)  # contrastive learning

    student = de_wide_resnet50_2(pretrained=False)
    DFS = DomainRelated_Feature_Selection()
    [source_teacher, bottleneck, student, DFS] = to_device([source_teacher, bottleneck, student, DFS], device)

    params = list(student.parameters()) + list(bottleneck.parameters()) + list(DFS.parameters())
    student_optimizer = torch.optim.AdamW(
        params, lr=args.lr_s, betas=(0.9, 0.999), weight_decay=1e-5)
    target_teacher_optimizer = torch.optim.AdamW(
        list(target_teacher.parameters()),
        lr=1e-4 if args._class_ == 'transistor' else args.lr_t,
        betas=(0.9, 0.999), weight_decay=1e-5)

    model = UniNet(args, source_teacher, target_teacher, bottleneck, student, DFS=DFS)
    model = model.to(device)

    # total_params = count_parameters(model)
    # print("Number of parameter: %.2fM" % (total_params/1e6))

    best_sample_level_auroc, best_pixel_level_auroc, best_pixel_level_aupro = 0.0, 0.0, 0.0

    total_iters = 1000
    if dataset_name == "ISIC2018":
        total_iters = 2000
    it = 0
    early_stopping = EarlyStopping(patience=3, verbose=False)

    # ---------------------------------------------training-----------------------------------------------
    for epoch in range(args.epochs):
        wandb_log_step = epoch  # epoch 단위 로깅 기준
        model.train_or_eval(type='train')
        loss_list = []
        for sample in train_dataloader:  # 배치 단위
            img = sample[0][0].to(device) if dataset_name == "MVTec 3D-AD" else sample[0].to(device)
            loss = model(img, stop_gradient=dataset_name in ["APTOS", "ISIC2018", "OCT2017"])
            student_optimizer.zero_grad()
            target_teacher_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(params, 0.5)
            student_optimizer.step()
            target_teacher_optimizer.step()
            loss_list.append(loss.item())

            # ------------------------------------eval medical AD-------------------------------------------
            if dataset_name in ["APTOS", "ISIC2018", "OCT2017"]:
                if (it + 1) % 250 == 0:
                    print('iters: {}/{}, loss:{:.4f}'.format(it + 1, total_iters, np.mean(loss_list)))

                    modules_list = [model.teacher.target_teacher, model.bottleneck.bottleneck, model.student.student_decoder, DFS]
                    auroc, f1, acc = evaluation_mediAD(args, model, test_dataloader, device)
                    print('Auroc: {:.2f}, f1: {:.2f}, acc: {:.2f}'.format(auroc, f1, acc))
                    wandb.log({
                        "auroc": auroc,
                        "f1_score": f1,
                        "accuracy": acc
                    }, step=it)
                    if best_sample_level_auroc < auroc:
                        best_sample_level_auroc = auroc
                        save_weights(modules_list, ckpt_path, "BEST_I_ROC") if args.is_saved else None
                    model.train_or_eval(type='train')
                it += 1
                if it > total_iters:
                    return

        # ------------------------------------eval industrial and video-------------------------------------

        if dataset_name in ['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA", 'ped2']:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, args.epochs, np.mean(loss_list)))
            ################### wandb log ###################
            wandb.log({"epoch_loss": np.mean(loss_list)}, step=wandb_log_step)

        modules_list = [model.teacher.target_teacher, model.bottleneck.bottleneck, model.student.student_decoder, DFS]
        is_best_sample_auroc = False
        if (epoch + 1) % 10 == 0 and args.domain in ['industrial', 'video']:

            if dataset_name in ['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA"]:
                # evaluation
                pixel_level_auroc, sample_level_auroc, pixel_level_aupro = evaluation_indusAD(args, model, test_dataloader, device)
                print('Sample Auroc: {:.1f}, Pixel Auroc: {:.1f}, Pixel Aupro: {:.1f}'.format(sample_level_auroc, pixel_level_auroc,
                                                                                                    pixel_level_aupro))
                ############ wandb log ############
                wandb.log({
                    "sample_auroc": sample_level_auroc,
                    "pixel_auroc": pixel_level_auroc,
                    "pixel_aupro": pixel_level_aupro
                }, step=wandb_log_step)
                if best_sample_level_auroc < sample_level_auroc:
                    best_sample_level_auroc = sample_level_auroc
                    is_best_sample_auroc = True

                if best_pixel_level_auroc < pixel_level_auroc:
                    best_pixel_level_auroc = pixel_level_auroc
                    save_weights(modules_list, ckpt_path, "BEST_P_ROC") if args.is_saved else None

                if (is_best_sample_auroc and best_pixel_level_aupro == pixel_level_aupro) or best_pixel_level_aupro < pixel_level_aupro:
                    best_pixel_level_aupro = pixel_level_aupro
                    print('saved')
                    save_weights(modules_list, ckpt_path, "BEST_P_PRO") if args.is_saved else None
                    ########## wandb save ############
                    wandb.save(os.path.join(ckpt_path, "BEST_P_PRO.pth"))

                print(f"MAX I_ROC: {best_sample_level_auroc:.1f}, MAX P_ROC: {best_pixel_level_auroc:.1f}, MAX P_PRO: {best_pixel_level_aupro:.1f}")
                early_stopping(pixel_level_aupro)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            else:
                test_folder = 'video/ped2/testing/frames'
                auroc = evaluation_video(args, model, test_folder, test_dataloader, device)
                print('Auroc: {:.2f}'.format(auroc))
