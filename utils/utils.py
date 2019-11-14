import os
import torch
import pickle
import numpy as np
import os.path as osp


def get_roi(rect, shape, margin_scale=0.2):
    y_min, y_max = rect.top(), rect.bottom()
    x_min, x_max = rect.left(), rect.right()

    delta_y = abs(y_max - y_min) * margin_scale
    delta_x = abs(x_max - x_min) * margin_scale

    new_y_min, new_y_max = max(0, y_min - delta_y), min(shape[0], y_max + delta_y)
    new_x_min, new_x_max = max(0, x_min - delta_x), min(shape[1], x_max + delta_x)
    # return [new_y_min, new_y_max, new_x_min, new_x_max]
    return [new_x_min, new_y_min, new_x_max, new_y_max]


def get_final_roi(pts, shape, margin_scale=0.2, do_scaling=False):
    pts = pts[:2]
    pts = pts.T
    left_jaw = pts[0]
    right_jaw = pts[16]
    left_eyebrow = 0.5 * (pts[18] + pts[19])
    right_eyebrow = 0.5 * (pts[23] + pts[24])
    nose = pts[34]

    min_x = left_jaw[0]
    max_x = right_jaw[0]

    min_y = min(left_eyebrow[1], right_eyebrow[1])
    max_y = nose[1]

    if do_scaling:
        delta_y = abs(max_y - min_y) * margin_scale
        min_y, max_y = max(0, min_y - delta_y), min(shape[0], max_y + delta_y)

    return list(map(int, [min_x, max_x, min_y, max_y]))


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)


def mkdir(d):
    """only works on *nix system"""
    if not os.path.isdir(d) and not os.path.exists(d):
        os.system('mkdir -p {}'.format(d))


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def _dump(wfp, obj):
    suffix = _get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':
        pickle.dump(obj, open(wfp, 'wb'))
    else:
        raise Exception('Unknown Type: {}'.format(suffix))


def _load_tensor(fp, mode='cpu'):
    if mode.lower() == 'cpu':
        return torch.from_numpy(_load(fp))
    elif mode.lower() == 'gpu':
        return torch.from_numpy(_load(fp)).cuda()


keypoints = _load('../config_stuff/keypoints_sim.npy')
w_shp = _load('../config_stuff/w_shp_sim.npy')
w_exp = _load('../config_stuff/w_exp_sim.npy')
meta = _load('../config_stuff/param_whitening.pkl')

param_mean = meta.get('param_mean')
param_std = meta.get('param_std')
u_shp = _load('../config_stuff/u_shp.npy')
u_exp = _load('../config_stuff/u_exp.npy')
u = u_shp + u_exp
w = np.concatenate((w_shp, w_exp), axis=1)
w_base = w[keypoints]
w_norm = np.linalg.norm(w, axis=0)
w_base_norm = np.linalg.norm(w_base, axis=0)

dim = w_shp.shape[0] // 3
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]
std_size = 120
