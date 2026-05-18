import copy
import os
from typing import Dict

import cv2
import numpy as np
import torch

from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix


class FileStorage:
    def __init__(self, filename: str, is_write: bool = False):
        self.is_write = is_write
        if is_write:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            if cv2 is None:
                raise RuntimeError('OpenCV (cv2) is required to read EasyVolcap camera files.')
            if not os.path.isfile(filename):
                raise FileNotFoundError(filename)
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        if not hasattr(self, 'fs') or self.fs is None:
            return
        if self.is_write:
            self.fs.close()
        else:
            self.fs.release()
        self.fs = None

    def close(self):
        self.__del__()

    def _write(self, out: str):
        self.fs.write(out + '\r\n')

    def write(self, key: str, value, dt: str = 'mat'):
        if dt == 'mat':
            value = np.asarray(value)
            self._write(f'{key}: !!opencv-matrix')
            self._write(f'  rows: {value.shape[0]}')
            self._write(f'  cols: {value.shape[1]}')
            self._write('  dt: d')
            flat = ', '.join([f'{i:.10f}' for i in value.reshape(-1)])
            self._write(f'  data: [{flat}]')
        elif dt == 'list':
            self._write(f'{key}:')
            for elem in value:
                self._write(f'  - "{elem}"')
        elif dt == 'real':
            if isinstance(value, np.ndarray):
                value = value.item()
            self._write(f'{key}: {value:.10f}')
        else:
            raise NotImplementedError(dt)

    def read(self, key: str, dt: str = 'mat'):
        node = self.fs.getNode(key)
        if node.empty():
            return None

        if dt == 'mat':
            return node.mat()
        if dt == 'list':
            values = []
            for i in range(node.size()):
                val = node.at(i).string()
                if val == '':
                    val = str(int(node.at(i).real()))
                if val != 'none':
                    values.append(val)
            return values
        if dt == 'real':
            return node.real()
        raise NotImplementedError(dt)


def _format_distortion(dist) -> np.ndarray:
    if dist is None:
        return np.zeros((5, 1), dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    if dist.size == 0:
        return np.zeros((5, 1), dtype=np.float64)
    if dist.size == 4:
        dist = np.concatenate([dist, np.zeros(1, dtype=np.float64)])
    elif dist.size != 5:
        raise ValueError(f'Unexpected distortion size: {dist.size}')
    return dist.reshape(5, 1)


def write_camera(cameras: Dict, path: str, intri_name: str = '', extri_name: str = ''):
    os.makedirs(path, exist_ok=True)
    if not intri_name or not extri_name:
        intri_name = os.path.join(path, 'intri.yml')
        extri_name = os.path.join(path, 'extri.yml')

    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)

    cam_names = [key.split('.')[0] for key in cameras.keys()]
    intri.write('names', cam_names, 'list')
    extri.write('names', cam_names, 'list')

    for key_, val in cameras.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]

        K = val['K']
        intri.write(f'K_{key}', K)
        if 'H' in val:
            intri.write(f'H_{key}', val['H'], 'real')
        if 'W' in val:
            intri.write(f'W_{key}', val['W'], 'real')

        D = val.get('D', val.get('dist', np.zeros((5, 1), dtype=np.float64)))
        D = _format_distortion(D)
        intri.write(f'D_{key}', D)

        R = val['R']
        T = val['T'].reshape(3, 1)
        if cv2 is not None:
            Rvec = cv2.Rodrigues(R)[0]
            extri.write(f'R_{key}', Rvec)
        extri.write(f'Rot_{key}', R)
        extri.write(f'T_{key}', T)


def read_camera(path: str, intri_name: str = '', extri_name: str = '') -> Dict:
    if not intri_name or not extri_name:
        intri_name = os.path.join(path, 'intri.yml')
        extri_name = os.path.join(path, 'extri.yml')

    intri = FileStorage(intri_name, is_write=False)
    extri = FileStorage(extri_name, is_write=False)
    cam_names = intri.read('names', dt='list')
    if not cam_names:
        cam_names = extri.read('names', dt='list')
    if not cam_names:
        raise ValueError(f'No camera names found in {intri_name} or {extri_name}')

    cameras = {}
    for name in cam_names:
        key = str(name).split('.')[0]
        K = intri.read(f'K_{key}')
        T = extri.read(f'T_{key}')
        if K is None:
            raise KeyError(f'Missing K_{key} in {intri_name}')
        if T is None:
            raise KeyError(f'Missing T_{key} in {extri_name}')

        Rvec = extri.read(f'R_{key}')
        if Rvec is not None:
            if cv2 is None:
                raise RuntimeError(f'OpenCV (cv2) is required to decode R_{key} from {extri_name}.')
            R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read(f'Rot_{key}')
            if R is None:
                raise KeyError(f'Missing both R_{key} and Rot_{key} in {extri_name}')

        D = intri.read(f'D_{key}')
        if D is None:
            D = intri.read(f'dist_{key}')

        cameras[key] = {
            'K': np.asarray(K, dtype=np.float64).reshape(3, 3),
            'R': np.asarray(R, dtype=np.float64).reshape(3, 3),
            'T': np.asarray(T, dtype=np.float64).reshape(3, 1),
            'H': int(intri.read(f'H_{key}', dt='real') or -1),
            'W': int(intri.read(f'W_{key}', dt='real') or -1),
            'D': _format_distortion(D),
        }

    intri.close()
    extri.close()
    return cameras


def load_traj_from_easyvolcap(path: str, template_cameras):
    if len(template_cameras) == 0:
        raise ValueError('template_cameras must contain at least one camera.')

    cameras = read_camera(path)
    template_cam = template_cameras[0]
    if torch.is_tensor(template_cam.world_view_transform):
        device = template_cam.world_view_transform.device
    else:
        device = torch.device('cuda')

    cam_traj = []
    for i, (_, cam_dict) in enumerate(cameras.items()):
        cam = copy.deepcopy(template_cam)
        W = int(cam_dict['W']) if int(cam_dict['W']) > 0 else int(cam.image_width)
        H = int(cam_dict['H']) if int(cam_dict['H']) > 0 else int(cam.image_height)

        K = cam_dict['K']
        fx = float(K[0, 0])
        fy = float(K[1, 1])

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = cam_dict['R'].astype(np.float32)
        w2c[:3, 3] = cam_dict['T'].reshape(3).astype(np.float32)

        cam.image_width = W
        cam.image_height = H
        cam.FoVx = focal2fov(fx, W)
        cam.FoVy = focal2fov(fy, H)
        cam.world_view_transform = torch.from_numpy(w2c.T).float().to(device)
        cam.projection_matrix = (
            getProjectionMatrix(
                znear=cam.znear,
                zfar=cam.zfar,
                fovX=cam.FoVx,
                fovY=cam.FoVy,
            )
            .transpose(0, 1)
            .to(device)
        )
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        cam.uid = i
        cam_traj.append(cam)

    return cam_traj


def save_traj_to_easyvolcap(cam_traj, out_dir: str, cam_digit: int = 6):
    cameras = {}
    for i, cam in enumerate(cam_traj):
        W = int(cam.image_width)
        H = int(cam.image_height)
        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        cx = 0.5 * W
        cy = 0.5 * H

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        if torch.is_tensor(cam.world_view_transform):
            w2c = cam.world_view_transform.detach().cpu().numpy().T
        else:
            w2c = np.asarray(cam.world_view_transform).T
        w2c = w2c.astype(np.float64)
        R = w2c[:3, :3]
        T = w2c[:3, 3:4]

        cameras[f'{i:0{cam_digit}d}'] = {
            'K': K,
            'R': R,
            'T': T,
            'H': H,
            'W': W,
            'D': np.zeros((5, 1), dtype=np.float64),
        }

    write_camera(cameras, out_dir)
