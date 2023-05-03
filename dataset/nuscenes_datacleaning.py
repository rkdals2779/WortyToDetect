import numpy as np
import json
import pickle as pkl
import os
import datacleaning_util as util
import cv2
from draw_polygon import Box3dPolygonDrawer


class NuscenesDataCleaning:
    def __init__(self, path='./data/nuscenes', split='train', threshold=0., reduce_rate=1.0):
        self.path = path
        self.split = split
        self.data = self.mmdet_pkl_load()
        self.threshold = threshold
        self.reduce_rate = reduce_rate
        self.img_shape = (900, 1600)
        self.draw_polygon = Box3dPolygonDrawer()

    def data_cleaning(self, cam='CAM_FRONT', vis=False, wait=0):
        for index in range(len(self.data['data_list'])):
            img_info = self.data['data_list'][index]['images'][cam]
            bboxes = self.data['data_list'][index]['cam_instances'][cam]
            bboxes = self.sort_by_depth(bboxes)
            poly_imgs = self.draw_polygon(bboxes, img_info['cam2img'], self.reduce_rate, self.img_shape)
            poly_map = self.get_poly_map(poly_imgs)
            visible_ratio = self.get_visible_ratio(poly_imgs, poly_map)
            bboxes_visible = self.filter_visible_boxes(bboxes, visible_ratio, self.threshold)
            if vis:
                self.visualization(cam, img_info, poly_map, wait)
            self.data['data_list'][index]['cam_instances'][cam] = bboxes_visible

    def visualization(self, cam, img_info, poly_map, wait=0):
        img = cv2.imread(os.path.join(self.path, 'samples', cam, img_info['img_path']))
        intrinsic = img_info['cam2img']
        N = np.max(poly_map)
        color = np.random.randint(-30, 30, (N, 3))
        img = img.astype(np.int32)
        for i in range(N):
            img[poly_map == i] += color[i]
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imshow('img', img)
        cv2.waitKey(wait)


    def sort_by_depth(self, bboxes):
        depth = [cam_data['depth'] for cam_data in bboxes]
        sorted_indices = np.argsort(depth)
        sorted_bboxes = [bboxes[ind] for ind in sorted_indices]
        return sorted_bboxes

    def get_poly_map(self, poly_imgs):
        map = np.zeros(self.img_shape[0] * self.img_shape[1])
        poly_imgs = poly_imgs.reshape(-1, 900 * 1600)
        for i, img in enumerate(poly_imgs):
            map[img == 255] = i + 1
        return map

    def get_visible_ratio(self, poly_imgs, poly_map):
        poly_imgs = poly_imgs.reshape(-1, 900*1600)
        area_ratio = [0] * len(poly_imgs)
        instance_sum = np.sum(poly_imgs, axis=-1) / 255
        for i in range(len(poly_imgs)):
            instance_map = len(np.where(poly_map == i + 1)[0])
            area_ratio[i] = instance_map / (instance_sum[i] + 1e-7)
        return np.array(area_ratio)

    def filter_visible_boxes(self, bboxes, visible_ratio, threshold):
        filter_indicies = np.where(visible_ratio > threshold)[0]
        filtered_bboxes = [bboxes[i] for i in filter_indicies]
        return filtered_bboxes

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

    # def visualization(self, cam, index, filter_idx, wait=0):
    #     img = cv2.imread(os.path.join(self.path, 'samples',
    #                                   cam, self.data['data_list'][index]['images'][cam]['img_path']))
    #     intrinsic = self.get_intrinsic(cam, index)
    #     bboxes3d_depth = self.get_cam_instances_bbox3d_depth(cam, index)
    #     if len(bboxes3d_depth):
    #         for i, box3d in enumerate(bboxes3d_depth[:, :7]):
    #             box3d[1] += box3d[4] / 2
    #             if i in filter_idx:
    #                 color = (0, 255, 0)
    #             else:
    #                 color = (0, 0, 255)
    #             img = util.draw_box3d_image(box3d, img, intrinsic, color)
    #     cv2.imshow('img2', img)
    #     cv2.waitKey(wait)

    def save_data_as_pkl(self):
        with open("data.pickle", "wb") as fw:
            pkl.dump(self.data, fw)


def main():
    nu = NuscenesDataCleaning(path='/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a1/datasets/nuscenes',
                              threshold=0.6, reduce_rate=0.9)
    num = len(nu.data['data_list'])
    for i in range(num):
        print('image num: ', i)
        nu.data_cleaning('CAM_FRONT',  vis=True)
    # nu.save_data_as_pkl()
    print()


if __name__ == '__main__':
    main()


def masking(image, polymap):
    """
    image: (H,W,3)
    polymap: (H,W)
    """
    N = np.max(polymap)
    color = np.random.randint(-30, 30, (N, 3))
    image = image.astype(np.int32)
    for i in range(N):
        image[polymap == i] += color[i]
    image = np.clip(image, 0, 255).astype(np.uint8)

    # draw 3d boxes
