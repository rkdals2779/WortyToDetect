import numpy as np
import cv2
from scipy.spatial import ConvexHull


def draw_box3d_image(box3d, img, intrinsic, color=(0, 255, 0)):
    corners_3d_cam2 = convert_point_to_corners(box3d)
    corners_2d = project_bbox3d_to_2d(corners_3d_cam2, intrinsic)
    corners_2d = corners_2d.astype(np.int32)

    for i in range(4):
        pt1 = (corners_2d[i][0], corners_2d[i][1])
        pt2 = (corners_2d[i + 4][0], corners_2d[i + 4][1])
        img = cv2.line(img, pt1, pt2, color, 2)

        pt1 = (corners_2d[i][0], corners_2d[i][1])
        pt2 = (corners_2d[(i + 1) % 4][0], corners_2d[(i + 1) % 4][1])
        img = cv2.line(img, pt1, pt2, color, 2)

        pt1 = (corners_2d[i + 4][0], corners_2d[i + 4][1])
        pt2 = (corners_2d[(i + 1) % 4 + 4][0], corners_2d[(i + 1) % 4 + 4][1])
        img = cv2.line(img, pt1, pt2, color, 2)
    return img


def convert_point_to_corners(box3d):
    x, y, z, l, h, w, yaw = box3d
    r_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                      [0, 1, 0],
                      [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners_3d_cam2 = np.dot(r_yaw, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def project_bbox3d_to_2d(bbox3d, intrinsic):
    pj_mat = np.hstack((intrinsic, np.zeros((3, 1))))
    pts_3d_hom = np.hstack((bbox3d.T, np.ones((bbox3d.T.shape[0], 1))))
    pts_2d_hom = np.dot(pts_3d_hom, pj_mat.T)
    pts_2d = pts_2d_hom[:, :2] / np.abs(pts_2d_hom[:, 2:3])
    return pts_2d


def get_convex_edge_points(pj_pts):
    hull = ConvexHull(pj_pts)
    edge_points = hull.vertices
    edge_pts = [pj_pts[i] for i in edge_points]
    return np.round(np.array(edge_pts)).astype(np.int32)


def fill_poly_to_img(edge_pts, img_shape):
    img = np.zeros(img_shape)
    img = cv2.fillConvexPoly(img, edge_pts, 255)
    return img


def center_calc(polygon):
    center = np.sum(polygon, axis=0)/6
    return np.round(center).astype(np.int32)


def reduce_polygon(polygon, rate):
    center = center_calc(polygon)
    w = polygon - center
    reduced_poly = polygon - w * (1-rate)
    return np.round(reduced_poly).astype(np.int32)

