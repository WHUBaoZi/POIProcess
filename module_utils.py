import numpy as np
import cv2
import os
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import scipy

from copy import deepcopy
from projection import surface_projection

def vis_img(name, im):
    ratiox = 1200/int(im.shape[0])
    ratioy = 1200/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    #cv2.moveWindow(name,0,0)
    if im.max() > 1:
        im = im/255.
    cv2.imshow(name,im)
    if name != 'mask':
        cv2.waitKey()

def draw_keyps(keypoints, img):
        # if len(keypoints) > 1:
        img = img.copy()
        for keyps in keypoints:
            for point in keyps:
                point = (int(point[0]), int(point[1]))
                img = cv2.circle(img, point, 3, (0, 255, 255), -1)
                # self.vis_img('keyp', img)
        return img

def draw_bbox(bbox, img, color=(255,255,0), thickness=5, win_location=[0,0]):
    bbox = np.array(bbox).reshape(2,2) - np.array(win_location) # draw img patch
    # img = img.copy()
    img = cv2.rectangle(img, tuple(np.array(bbox[0], dtype=np.int64)), tuple(np.array(bbox[1], dtype=np.int64)), color, thickness)
    return img

def draw_bboxes(bboxes, img, person_ids = None, thickness=5):
    for i, bbox in enumerate(bboxes):
        img = img = cv2.rectangle(img, tuple(np.array(bbox[0], dtype=np.int64)), tuple(np.array(bbox[1], dtype=np.int64)), color=(255,255,0), thickness=thickness)
        if person_ids:
            index = person_ids[i]
            img = cv2.putText(img, '%s' %index, (int((bbox[0][0]+bbox[1][0])/2), int((bbox[0][1] + bbox[1][1])/2)), cv2.FONT_HERSHEY_PLAIN, 5.0, (0,0,0), thickness=thickness)
    return img

def draw_bboxes_batch(bboxes, img, thickness=5, win_location=[0,0], color=(255,255,0)):
    bboxes =np.array(bboxes) - np.array(win_location) # draw img patch
    tl = bboxes[:,0]
    br = bboxes[:,1]
    tr = np.array([br[:,0], tl[:,1]]).T
    bl = np.array([tl[:,0], br[:,1]]).T

    contours = np.array([tl, tr, br, bl]).transpose(1,0,2)
    # otherwise, error: npoints > 0 in function 'cv::drawContours'
    contours = np.array(contours.tolist()).astype(np.int32)
    img = cv2.drawContours(img, contours, -1, color, thickness)

    return img

def draw_keyp(img, joints, color=None, format='coco17', thickness=3):
    skeletons = {'coco17':[[0,1],[1,3],[0,2],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],    [14,16],[11,12]],
            'halpe':[[0,1],[1,3],[0,2],[2,4],[5,18],[6,18],[18,17],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16],[11,19],[19,12],[18,19],[15,24],[15,20],[20,22],[16,25],[16,21],[21,23]],
            'MHHI':[[0,1],[1,2],[3,4],[4,5],[0,6],[3,6],[6,13],[13,7],[13,10],[7,8],[8,9],[10,11],[11,12]],
            'Simple_SMPL':[[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,10],[10,11],[11,12],[8,13],[13,14],[14,15]],
            'LSP':[[0,1],[1,2],[2,3],[5,4],[4,3],[3,9],[9,8],[8,2],[6,7],[7,8],[9,10],[10,11]],
            }
    colors = {'coco17':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127)],
                'halpe':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), ],
                'MHHI':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127)],
                'Simple_SMPL':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127)],
                'LSP':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127)]}

    if joints.shape[1] == 3:
        confidence = joints[:,2]
    else:
        confidence = np.ones((joints.shape[0], 1))
    joints = joints[:,:2].astype(np.int64)
    for bone, c in zip(skeletons[format], colors[format]):
        if color is not None:
            c = color
        # c = (0,255,255)
        if confidence[bone[0]] > 0.1 and confidence[bone[1]] > 0.1:
            # pass
            img = cv2.line(img, tuple(joints[bone[0]]), tuple(joints[bone[1]]), c, thickness=int(thickness))
    
    for p in joints:
        img = cv2.circle(img, tuple(p), int(thickness * 5/3), c, -1)
        # vis_img('img', img)
    return img

def get_halpe(smpl_vert):
    halpe_regressor = np.load('models/smpl/J_regressor_halpe.npy')
    halpe_joints_3d = np.dot(halpe_regressor, smpl_vert)
    return halpe_joints_3d

def calculate_scale(flags, step, scale):
    # zoom in 
    if flags > 0:
        scale += step
        if scale > 15.0: # max scale
            scale = 15.0
    # zoom out
    else:
        scale -= step
        if scale < 1.0:
            scale = 1.0
    return scale

def check_window_location(img_wh, img_patch_wh, win_location):
    # ensure the location of window is inside the full image
    if win_location[0] < 0:
        win_location[0] = 0
    elif win_location[0] + img_patch_wh[0] > img_wh[0]:
        win_location[0] = int(img_wh[0] - img_patch_wh[0]) 
    
    if win_location[1] < 0:
        win_location[1] = 0
    elif win_location[1] + img_patch_wh[1] > img_wh[1]:
        win_location[1] = int(img_wh[1] - img_patch_wh[1]) 
    
    return win_location

def convert2ori_img(bbox, win_location):
    bbox = np.array(bbox).reshape(-1,2)
    win_location = np.array(win_location)

    bbox = (bbox + win_location) 

    bbox = bbox.reshape(-1).astype(np.int32).tolist()
    win_location = win_location.tolist()
    return bbox

def calculate_bbox(verts, intri, img, smpl):
    verts_batch = verts.reshape(-1,3)
    verts_batch_2d, _ = surface_projection(verts_batch, smpl.faces, None, np.eye(4), intri, img, viz=False)
    verts_2d = verts_batch_2d.reshape(len(verts), -1, 2)
    
    xmin = np.maximum(0, np.min(verts_2d[:,:,0], axis=1)) 
    ymin = np.maximum(0, np.min(verts_2d[:,:,1], axis=1))  
    xmax = np.minimum(img.shape[1], np.max(verts_2d[:,:,0], axis=1))  
    ymax = np.minimum(img.shape[0], np.max(verts_2d[:,:,1], axis=1))
    
    bbox = np.array([[xmin, ymin], [xmax, ymax]]).transpose(2,0,1)  
    
    return bbox

def IOUcheck(pose1, pose2):
    x1min, y1min, x1max, y1max = pose1[0], pose1[1], pose1[2], pose1[3]
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    x2min, y2min, x2max, y2max = pose2[0], pose2[1], pose2[2], pose2[3]
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)
    
    xmin = max(x1min, x2min) 
    ymin = max(y1min, y2min)  
    xmax = min(x1max, x2max)  
    ymax = min(y1max, y2max)  
  
    inter_h = max(ymax - ymin + 1., 0.)
    inter_w = max(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
   
    union = s1 + s2 - intersection
    iou = intersection / union
    return iou

def IOUcheck_batch(poses1, poses2):
    poses1 = np.array(poses1).reshape(-1,4)
    poses2 = np.array(poses2).reshape(-1,4)

    x1min, y1min, x1max, y1max = poses1[:,0], poses1[:,1], poses1[:,2], poses1[:,3]
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    x2min, y2min, x2max, y2max = poses2[:,0], poses2[:,1], poses2[:,2], poses2[:,3]
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    xmin = np.maximum(x1min.reshape(len(x1min), -1), x2min.reshape(-1, len(x2min))) 
    ymin = np.maximum(y1min.reshape(len(y1min), -1), y2min.reshape(-1, len(y2min)))  
    xmax = np.minimum(x1max.reshape(len(x1max), -1), x2max.reshape(-1, len(x2max)))  
    ymax = np.minimum(y1max.reshape(len(y1max), -1), y2max.reshape(-1, len(y2max)))  

    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    ious = intersection / union
    return ious.tolist()


def get_verts(poses, shapes, trans, smpl):
    # get smpl verts
    pose = torch.from_numpy(np.array(poses, dtype=np.float32).reshape(-1,72))
    shape = torch.from_numpy(np.array(shapes, dtype=np.float32).reshape(-1,10))
    trans = torch.from_numpy(np.array(trans, dtype=np.float32).reshape(-1,3))
    smpl_verts, joints = smpl(shape, pose, trans)
    smpl_verts = smpl_verts.detach().cpu().numpy()  
    joints = joints.detach().cpu().numpy()  

    return smpl_verts, joints

def render_img(smpl_verts, img, smpl, intri=None, win_location=[0,0]):
    # return real intri, and intri_copy used to render img or img_patch(needs to transform intri value)
    if intri is None:
        intri = np.eye(3)
        focal = np.sqrt((img.shape[0]**2 + img.shape[1]**2))
        intri[0][0] = focal
        intri[1][1] = focal
        intri[0][2] = img.shape[1] / 2
        intri[1][2] = img.shape[0] / 2
        intri_copy = deepcopy(intri)
    else:
        intri_copy = deepcopy(intri)
        intri_copy[0][2] -= win_location[0]
        intri_copy[1][2] -= win_location[1]
    
    render = Renderer(resolution=(img.shape[1], img.shape[0]))
    img_render = img.copy()
    img_render = render.render_multiperson(smpl_verts, smpl.faces, np.eye(3), np.zeros((3,)), intri_copy, img_render, viz=False)

    img_render = img_render.astype(np.uint8)

    render.renderer.delete()

    return img_render, intri

def local_render_img(smpl_verts, bboxes, img, img_render, win_location, img_patch_wh, intri, smpl):
    # get render people id according to bbox and window location
    render_id = calculate_window_iou_id(win_location, img_patch_wh, bboxes)
    render_verts = smpl_verts[render_id]
    
    # only render people in the window
    img_patch = crop_img_patch(img, win_location, img_patch_wh)
    img_patch_render, _ = render_img(render_verts, img_patch, smpl, intri, win_location)
    # vis_img('img_patch_render', img_patch_render)
    img_render = paste_img_patch(img_render, img_patch_render, win_location, img_patch_wh)

    return img_render

def calculate_window_iou_id(win_location, img_patch_wh, bboxes):
    window = [win_location[0], win_location[1], win_location[0] + img_patch_wh[0], win_location[1] + img_patch_wh[1]]
    ious = IOUcheck_batch(window, bboxes)[0]
    
    ious = np.array(ious)
    iou_id = np.where(ious > 0)
    return iou_id

def crop_img_patch(img, win_location, img_patch_wh):
    img_patch = img[win_location[1]:win_location[1] + img_patch_wh[1],        \
                win_location[0]:win_location[0] + img_patch_wh[0]]
    return img_patch

def paste_img_patch(img, img_patch, win_location, img_patch_wh):
    img[win_location[1]:win_location[1] + img_patch_wh[1],        \
                win_location[0]:win_location[0] + img_patch_wh[0]] = img_patch
    return img

def local_draw_bbox(bbox, img, win_location, img_patch_wh, color, thickness):
    img_patch = crop_img_patch(img, win_location, img_patch_wh) 
    img_bbox_patch = draw_bbox(bbox, img_patch.copy(), color=color, thickness=thickness, win_location=win_location)
    # img_bbox = paste_img_patch(img.copy(), img_bbox_patch, win_location, img_patch_wh)
    return img_bbox_patch

def local_draw_bboxes(bboxes, img_bbox, img_render, win_location, img_patch_wh, thickness, color=(255,255,0)):
    bboxes_id = calculate_window_iou_id(win_location, img_patch_wh, bboxes)
    bboxes_draw = bboxes[bboxes_id]

    # only draw bboxes in the window
    img_patch = crop_img_patch(img_render, win_location, img_patch_wh) 
    img_bboxes_patch = draw_bboxes_batch(bboxes_draw, img_patch.copy(), thickness=thickness, win_location=win_location, color=color)
    img_bbox = paste_img_patch(img_bbox, img_bboxes_patch, win_location, img_patch_wh)
    return img_bbox

def joints_dist_norm(joints_2d, bboxes, ious_id):
    # joints distance
    joints_2d_dis = joints_2d[ious_id[:,0]] - joints_2d[ious_id[:,1]]
    joints_2d_dis = np.linalg.norm(joints_2d_dis, axis=2)
    joints_2d_dis = np.sum(joints_2d_dis, axis=1)

    # bbox size
    bboxes = np.array(deepcopy(bboxes)).reshape(-1,4)
    bboxes = bboxes[ious_id[:,0]]
    xmin, ymin, xmax, ymax = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
    bboxes_s = (ymax - ymin + 1.) * (xmax - xmin + 1.)

    # normalize
    joints_2d_dis_norm = joints_2d_dis / bboxes_s
    
    return joints_2d_dis_norm
