
import numpy as np
import cv2
import pandas as pd
import os
from constants import *
from unet_pretrained import get_pretrained_unet, convert_grayscale_batch_to_rgb
from utils import compute_dice_coef, print_and_log

def compute_train_sets_ds(X_train, y_train, labeled_index, unlabeled_index, weights_path, iteration, nb_next_sample,
                          logfile, nb_pseudo=0, crf_configs=None, pre_ensemble=False):
    print_and_log(f"\n[Iteration {iteration}] Starting selection loop...", logfile)

    # Load pretrained model
    modelPredictions = get_pretrained_unet(input_shape=(img_rows, img_cols, 1))
    modelPredictions.load_weights(weights_path)

    # Preprocess for RGB model input
    X_input = convert_grayscale_batch_to_rgb(X_train[unlabeled_index])
    predictions = modelPredictions.predict(X_input, verbose=0)

    df = pd.DataFrame(unlabeled_index, columns=['unlabeled_index'])
    df['R-DSC'] = ''

    for index, uid in enumerate(unlabeled_index):
        pred_mask = predictions[index].squeeze()
        bin_mask = cv2.threshold(pred_mask, 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        gt_mask = y_train[uid].squeeze()
        dice = compute_dice_coef(gt_mask, bin_mask)
        df.loc[index, 'R-DSC'] = dice

        if index % 10 == 0:
            save_path = f"{global_path}{exp}_prediction/{uid}_prediction.png"
            cv2.imwrite(save_path, bin_mask * 255)

    df.to_csv(global_path + f'{exp}_ranks/predictions_iteration_{iteration}.csv')
    sorted_df = df.sort_values('R-DSC', ascending=DS_ASCEND)

    if random_inc:
        selected_indices = np.random.choice(unlabeled_index, nb_next_sample, replace=False)
    else:
        selected_indices = sorted_df['unlabeled_index'].iloc[:nb_next_sample].to_numpy()

    pseudo_rank = []

    # Prepare final labeled sets
    labeled_index = np.concatenate((labeled_index, selected_indices)).astype(int)

    X_labeled_train = X_train[labeled_index]
    y_labeled_train = y_train[labeled_index]

    print_and_log(f"✔️ Labeled: {len(labeled_index)}, Unlabeled: {len(unlabeled_index)}", logfile)

    # Update unlabeled index
    unlabeled_index = [uid for uid in unlabeled_index if uid not in selected_indices]

    return X_labeled_train, y_labeled_train, labeled_index, unlabeled_index
