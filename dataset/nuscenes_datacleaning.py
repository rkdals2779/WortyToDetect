import numpy as np
import json
import pickle as pkl
import os
import datacleaning_util as util
import cv2


class NuscenesDataCleaning:
    def __init__(self, path='./data/nuscenes', split='train'):
        self.path = path
        self.split = split
        self.data = self.mmdet_pkl_load()
        self.img_shape = (900, 1600)

    def mmdet_pkl_load(self):
        with open(os.path.join(self.path, f'nuscenes_infos_{self.split}.pkl'), 'rb') as fr:
            pkl_data = pkl.load(fr)
        return pkl_data

    def get_cam_instances_bbox3d_depth(self, cam='CAM_FRONT', index=0):
        cam_instances_bbox3d = []
        for cam_data in self.data['data_list'][index]['cam_instances'][cam]:
            cam_instances_bbox3d.append(cam_data['bbox_3d'] + [cam_data['depth']])
        return np.array(cam_instances_bbox3d)

    def get_intrinsic(self, cam='CAM_FRONT', index=0):
        return np.array(self.data['data_list'][index]['images'][cam]['cam2img'])

    def get_filled_polygon_sort_d(self, cam, index, img_shape, reduce_rate=1.0):
        bbox3d_and_depth = self.get_cam_instances_bbox3d_depth(cam, index)
        if len(bbox3d_and_depth) == 0:
            return [], []
        intrinsic = self.get_intrinsic(cam, index)
        fill_vec = []
        for box3d_depth in bbox3d_and_depth:
            box3d_depth[1] += box3d_depth[4]/2
            corners = util.convert_point_to_corners(box3d_depth[:7])
            pj_pts = util.project_bbox3d_to_2d(corners, intrinsic)
            edge = util.get_convex_edge_points(pj_pts)
            edge = util.reduce_polygon(edge, reduce_rate)
            filledge_img = util.fill_poly_to_img(edge, img_shape)
            fill_vec.append(filledge_img.reshape(-1))
        further_depth_arg = np.argsort(-bbox3d_and_depth[:, -1])
        fill_vec = np.take(fill_vec, further_depth_arg, axis=0)
        return fill_vec, further_depth_arg

    def get_area_ratio(self, cam, index, reduce_rate=1.0, vis=False, wait=0):
        map = np.zeros(self.img_shape[0]*self.img_shape[1])
        fill_imgs, further_d_arg = self.get_filled_polygon_sort_d(cam, index, self.img_shape, reduce_rate)
        area_ratio = [0] * len(further_d_arg)
        for i, img in enumerate(fill_imgs):
            map[img != 0] = i+1
        instance_sum = np.sum(fill_imgs, axis=-1) / 255
        for i, arg in enumerate(further_d_arg):
            instance_map = len(np.where(map == i+1)[0])
            area_ratio[arg] = instance_map/(instance_sum[i] + 1e-7)
        if vis:
            map = map.reshape(self.img_shape).astype(np.uint8) * 30
            map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
            cv2.imshow('img', map)
            cv2.waitKey(wait)
        return np.array(area_ratio), map, further_d_arg

    def highlight_cleared_objects_vis(self, map, cleared_objects, wait=0):
        map = map.reshape(self.img_shape).astype(np.uint8) * 30
        map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
        for i in cleared_objects:
            map = np.where(map == i+1, i+1, i+1)
        cv2.imshow('img', map)

    def visualization(self, cam, index, filter_idx, wait=0):
        img = cv2.imread(os.path.join(self.path, 'samples', cam, self.data['data_list'][index]['images'][cam]['img_path']))
        intrinsic = self.get_intrinsic(cam, index)
        bboxes3d_depth = self.get_cam_instances_bbox3d_depth(cam, index)
        if len(bboxes3d_depth):
            for i, box3d in enumerate(bboxes3d_depth[:, :7]):
                box3d[1] += box3d[4] / 2
                if i in filter_idx:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                img = util.draw_box3d_image(box3d, img, intrinsic, color)
        cv2.imshow('img2', img)
        cv2.waitKey(wait)

    def data_cleaning(self, cam='CAM_FRONT', index=0, threshold=0., reduce_rate=1.0, vis=False, wait=0):
        area_ratio, map, furd_d_arg = self.get_area_ratio(cam, index, reduce_rate, vis, wait)
        filter_idx = np.where(area_ratio > threshold)[0]
        if vis:
            self.visualization(cam, index, filter_idx, wait)
        converted_data = [self.data['data_list'][index]['cam_instances'][cam][i] for i in filter_idx]
        self.data['data_list'][index]['cam_instances'][cam] = converted_data

    def save_data_as_pkl(self):
        with open("data.pickle", "wb") as fw:
            pkl.dump(self.data, fw)


def main():
    nu = NuscenesDataCleaning(path='/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/datasets/nuscenes')
    num = len(nu.data['data_list'])
    for i in range(num):
        print('image num: ', i)
        nu.data_cleaning('CAM_FRONT', i, threshold=0.6, reduce_rate=0.9, vis=True)
    # nu.save_data_as_pkl()
    print()


if __name__ == '__main__':
    main()
