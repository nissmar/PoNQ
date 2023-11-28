import numpy as np
import joblib
import argparse
from eval_Template import get_cd_f1_nc

gt_dir = "/data/nmaruani/DATASETS/ABC/"
all_models = "src/eval/abc_ordered.txt"
nw_list = 'src/eval/not_watertight_ABC_test.txt'

def eval_normalization(x):
    return x/2.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='metrics on dataset')
    parser.add_argument('pred_dir', type=str, help='Path to shape folder.')
    args = parser.parse_args()
    pred_dir = args.pred_dir

    fin = open(nw_list, 'r')
    nw = [name.strip()[:-5] for name in fin.readlines()]
    fin.close()

    fin = open(all_models, 'r')
    obj_names = [name.strip()[:-5] for name in fin.readlines()]
    obj_names_old = obj_names[int(len(obj_names)*0.8):]
    obj_names = [e for e in obj_names_old if not (e in nw)]
    fin.close()

    obj_names_len = len(obj_names)

    # prepare list of names
    list_of_list_of_names = []
    true_idx = 0
    for idx in range(len(obj_names_old)):
        if not obj_names_old[idx] in nw:
            gt_obj_name = "{}/{}/model.obj".format(gt_dir, obj_names_old[idx])
            pred_obj_name = "{}/test_{}.obj".format(pred_dir, str(idx))
            list_of_list_of_names.append(
                [idx, gt_obj_name, pred_obj_name])
            true_idx += 1

    out = joblib.Parallel(n_jobs=-1)(joblib.delayed(get_cd_f1_nc)
                                     (name, 1, eval_normalization) for name in (list_of_list_of_names))
    out = np.array(out)
    save_name = pred_dir.split('/')
    if len(save_name[-1]) > 0:
        save_name = save_name[-1]
    else:
        save_name = save_name[-2]
    np.save('src/eval/results/results_{}.npy'.format(save_name), out)

    mean_scores = out.mean(0)
    print('CD (x 1e-5), F1, NC, ECD, EF1')
    print('{} & {:.3f}  &  {:.3f}  &  {:.3f} & {:.3f} & {:.3f}'.format(save_name,
                                                                       mean_scores[2]*1e5, mean_scores[3], mean_scores[4],  mean_scores[5]*1e2,  mean_scores[6]))
