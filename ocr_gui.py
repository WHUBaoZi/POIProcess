
from paddleocr import PaddleOCR
from FileLoaders import *
from module_utils import *
from projection import joint_projection_np
from poi_correct_utils import draw_ocr, update_text_vis

keyboardSet = 0
select_bbox = [0, 0, 0, 0]
added_box = [0, 0, 0, 0]
g_startPoint = [0, 0] # start point of bbox clicked by left button
select_bbox_flag = False
select_all_bbox_flag = False
add_box = False
selected_a_bbox_flag = False

wheel_step = 0.2
scale = 1.0
win_location = [0, 0] # location of window relative to original image  (x,y)
img_patch_wh = [0, 0] # img width and height after reszie

win_location_move = [0, 0] # location of window relative to original image after being dragged by the right button
r_click_position = [0, 0] # click position of right button relative to window


def onMouse(event, x, y, flags, param): 
    global select_bbox_flag
    global select_all_bbox_flag
    global select_bbox
    global g_startPoint
    
    global wheel_step
    global scale
    global win_location
    global img_patch_wh

    global win_location_move 
    global r_click_position
    global added_box
    global add_box
    global selected_a_bbox_flag


    # click left button
    if event == cv2.EVENT_LBUTTONDOWN and (not flags == cv2.EVENT_FLAG_CTRLKEY):
        g_startPoint[0] = x
        g_startPoint[1] = y

    # select bbox by left button
    elif event == cv2.EVENT_MOUSEMOVE and (flags == cv2.EVENT_FLAG_LBUTTON) and (not flags == cv2.EVENT_FLAG_CTRLKEY):
        select_bbox[0] = min(g_startPoint[0], x)
        select_bbox[1] = min(g_startPoint[1], y)
        select_bbox[2] = max(g_startPoint[0], x)
        select_bbox[3] = max(g_startPoint[1], y) 
        
        # convert the selected bbox to original img size 
        select_bbox = convert2ori_img(select_bbox, win_location)
        
    # up left button
    elif event == cv2.EVENT_LBUTTONUP and any(select_bbox) and (not flags == cv2.EVENT_FLAG_CTRLKEY):
        if add_box:
            selected_a_bbox_flag = True
            added_box = select_bbox
        select_bbox_flag = True
    
    # select all the bbox by pressing on ctrl
    elif flags == (cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON) and event == cv2.EVENT_LBUTTONDOWN:
        g_startPoint[0] = x
        g_startPoint[1] = y
    
    elif event == cv2.EVENT_MOUSEMOVE and (flags == cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_FLAG_CTRLKEY):
        select_bbox[0] = min(g_startPoint[0], x)
        select_bbox[1] = min(g_startPoint[1], y)
        select_bbox[2] = max(g_startPoint[0], x)
        select_bbox[3] = max(g_startPoint[1], y)

        select_bbox = convert2ori_img(select_bbox, win_location)
    
    elif event == cv2.EVENT_LBUTTONUP and any(select_bbox) and (flags == cv2.EVENT_FLAG_CTRLKEY):
        select_all_bbox_flag = True

    # zoom in or zoom out on the image
    elif event == cv2.EVENT_MOUSEWHEEL:  
        prev_scale = scale  # save previous scale
        scale = calculate_scale(flags, wheel_step, scale)

        img_wh = np.array([int(param[0]), int(param[1])])  # window width and height
        patch_ratio = 1 / scale
        img_patch_wh = (img_wh * patch_ratio).astype(np.int32)
        
        # location of window relative to full image after resize
        win_location = [int(win_location[0] + x * scale / prev_scale - x), int(win_location[1] + y * scale / prev_scale - y)]
        # check boundary condition
        win_location = check_window_location(img_wh, img_patch_wh, win_location)
        
    # drag the local image after zooming in on the image
    elif event == cv2.EVENT_RBUTTONDOWN and scale > 1.0:
        r_click_position = [x, y]
        win_location_move = deepcopy(win_location) # initial window location relative to full image before dragged
    
    # drag the local image
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON) and scale > 1.0:
        r_drag_position = [x, y]
        win_location[0] = win_location_move[0] + r_click_position[0] - r_drag_position[0]
        win_location[1] = win_location_move[1] + r_click_position[1] - r_drag_position[1]
        # check boundary condition
        img_wh = np.array([int(param[0]), int(param[1])])
        win_location = check_window_location(img_wh, img_patch_wh, win_location)  

def auto_select(person_files, smpl_verts, bboxes, img, intri, joints_3d, thres=10):
    # project for 2d joints
    joints_2d, _ = joint_projection_np(joints_3d.reshape(-1, 3), np.eye(4), intri, img, viz=False)
    joints_2d = joints_2d.reshape(len(joints_3d), -1, 2)

    # intersections of bboxes between all people pairwise
    ious = IOUcheck_batch(bboxes, bboxes)
    ious = np.array(ious)
    # diagonal id is the same person, and upper triangle and lower triangle are the same people pairs, which need to delete
    ious = np.triu(ious, k=1)
    # calculate for pairwise ids whose bboxes have intersections
    ious_id = np.argwhere(ious > 0)
    
    # delete the same people
    joints_2d_dis = joints_dist_norm(joints_2d, bboxes, ious_id)
    ious_idx = np.where(joints_2d_dis*1000 < thres)
    delete_id = ious_id[ious_idx][:, 1]
    delete_id = np.unique(delete_id)

    select_files = np.delete(np.array(person_files), delete_id, axis=0)
    select_verts = np.delete(smpl_verts, delete_id, axis=0)
    select_bbox = np.delete(np.array(bboxes), delete_id, axis=0)

    return select_files.tolist(), select_verts, select_bbox

def load_img_param(param_dir, img_name, img_dir, smpl, thres=10, scale=0.5):
    param_path = osp.join(param_dir, img_name)
    files = os.listdir(param_path)
    
    print(' load ori img...')
    img_file = osp.join(img_dir, img_name + '.jpg')
    img = cv2.imread(img_file)
    img = cv2.resize(img, None, fx=scale, fy=scale)

    # set thickness
    size = max(img.shape[0], img.shape[1])
    thickness = int(max(min( size / 400., 8), 1))

    # intri
    intri = np.eye(3)
    focal = np.sqrt((img.shape[0]**2 + img.shape[1]**2))
    intri[0][0] = focal
    intri[1][1] = focal
    intri[0][2] = img.shape[1] / 2
    intri[1][2] = img.shape[0] / 2
    
    poses = []
    shapes = []
    trans = []
    for file in files:
        person_file = osp.join(param_path, file)
        person_data = load_pkl(person_file)

        pose = person_data['pose']
        shape = person_data['betas']
        transl = person_data['trans']
        poses.append(pose)
        shapes.append(shape)
        trans.append(transl)
    
    smpl_verts, joints_3d = get_verts(poses, shapes, trans, smpl)
    bboxes = calculate_bbox(smpl_verts, intri, img, smpl)

    # initial img shape
    global img_patch_wh
    img_patch_wh = [img.shape[1], img.shape[0]]
    
    # original params for render
    print('render img...')
    ori_img_render, _ = render_img(smpl_verts, img, smpl)

    # auto select 
    files, smpl_verts, bboxes = auto_select(files, smpl_verts, bboxes, img, intri, joints_3d, thres=thres)

    print('render img after auto select...')
    img_render, _ = render_img(smpl_verts, img, smpl)
    print('draw bbox...')
    img_bbox = draw_bboxes_batch(bboxes, img_render.copy(), thickness=thickness)

    return files, smpl_verts, bboxes, img, img_render, img_bbox, thickness, intri, ori_img_render


def select_params(img_path, ocr, font_path, user_check=True, thres=10):
    img = cv2.imread(img_path)
    result = ocr.ocr(img_path, cls=True)

    success_save = True

    if result[0] is None:
        return None, None, None, None, False

    # load params
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    img_bbox, txt_img, im_show = draw_ocr(img.copy(), boxes, txts, scores, font_path=font_path)

    bboxes = np.array(boxes)
    bboxes = np.array([[box[:,0].min(), box[:,1].min(), box[:,0].max(), box[:,1].max()] for box in bboxes])
    bboxes = bboxes.reshape(-1, 2, 2).astype(np.float32)

    # set thickness
    size = max(img.shape[0], img.shape[1])
    thickness = int(max(min( size / 800., 8), 1))

    # create img window
    ratiox = 1200 / int(img.shape[0])
    ratioy = 1200 / int(img.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    # resize img if img is too large
    if img.shape[1] > 27000:
        resize_ratio = 27000 / img.shape[1]
    else:
        resize_ratio = 1.

    if user_check:
        cv2.namedWindow("image", 0)
        cv2.moveWindow("image", 0, 0)
        cv2.resizeWindow("image", int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        cv2.setMouseCallback("image", onMouse, [img.shape[1] * resize_ratio, img.shape[0] * resize_ratio])

        cv2.namedWindow("text", 0)
        cv2.resizeWindow("text", int(txt_img.shape[1] * ratio), int(txt_img.shape[0] * ratio))
        cv2.setMouseCallback("text", onMouse, [txt_img.shape[1] * resize_ratio, txt_img.shape[0] * resize_ratio])
        cv2.moveWindow("text", 1200, 0)
        # cv2.imshow("image", img_bbox)
        # cv2.waitKey()

        # initialize variables
        # initialize variables
        global select_bbox
        global select_bbox_flag
        global select_all_bbox_flag
        global scale
        global img_patch_wh
        global win_location
        global added_box
        global add_box
        global selected_a_bbox_flag

        bboxes = bboxes * resize_ratio
        img = cv2.resize(img, (0,0), fx=resize_ratio, fy=resize_ratio)
        img_bbox = cv2.resize(img_bbox, (0,0), fx=resize_ratio, fy=resize_ratio)
        txt_img = cv2.resize(txt_img, (0,0), fx=resize_ratio, fy=resize_ratio)

        img_patch_wh = [img.shape[1], img.shape[0]]
        img_render_copy = img_bbox.copy()
        show_img_selected = img_render_copy
        bboxes_copy = deepcopy(bboxes)

        keyboardSet = -1
        select_id = None
        selected_bbox_img = None
        show_flag = 0
        success_save = False
        vis_output_flag = 0
        all_render_flag = 0
        win_location_bbox = deepcopy(win_location)
        img_patch_wh_bbox = deepcopy(img_patch_wh)
        del_list = []


        while True:
            #calculate ious, highlight selected bbox
            if (select_bbox_flag or select_all_bbox_flag) and any(select_bbox):
                ious = IOUcheck_batch(select_bbox, bboxes_copy)[0] # many to many
                # only show the draw bbox which has intersections with people bbox
                if any(ious) and select_bbox_flag:
                    select_id = [ious.index(max(ious))]
                    selected_bbox_img = local_draw_bbox(np.array(bboxes_copy)[select_id], img_render_copy, win_location, img_patch_wh, color=(0,0,255), thickness=thickness)
                    win_location_bbox = win_location.copy()
                    img_patch_wh_bbox = img_patch_wh.copy()
                    show_img_selected = selected_bbox_img
                elif any(ious) and select_all_bbox_flag:
                    select_id = np.where(np.array(ious) > 0)
                    selected_bbox_img = local_draw_bboxes(np.array(bboxes_copy)[select_id], img_bbox.copy(), img_render_copy, win_location, img_patch_wh, thickness, color=(0,0,255))
                    show_img_selected = selected_bbox_img
                    win_location_bbox = win_location.copy()
                    img_patch_wh_bbox = img_patch_wh.copy()
                else:
                    show_img_selected = img_bbox
                    select_bbox_flag = False
                    select_all_bbox_flag = False
                select_bbox = [0,0,0,0]

            # has drawn the selected bbox and wait to the next action
            elif (select_bbox_flag or select_all_bbox_flag) and (not any(select_bbox)) and (selected_bbox_img is not None):
                show_img_selected = selected_bbox_img

            elif any(select_bbox):
                new_bbox_img = local_draw_bbox(select_bbox, img_bbox, win_location, img_patch_wh, color=(0,0,255), thickness=thickness)
                show_img_selected = new_bbox_img

            else:
                show_img_selected = img_bbox

            # update text
            if keyboardSet == ord('u') or keyboardSet == ord('U'):
                user_input = input("Enter text: ")

                # update text
                txts[select_id[0]] = user_input
                scores[select_id[0]] = 1.0

            # selected person is wrong
            elif keyboardSet == ord('w') or keyboardSet == ord('W'):
                select_bbox_flag = False
                select_all_bbox_flag = False
                keyboardSet = -1
                # vis_img('show img_bbox', img_bbox)
                continue

            # selected person is right
            elif keyboardSet == ord('d') or keyboardSet == ord('D'):  # delete
                if select_id is not None:
                    # 确保 select_id 是整数列表
                    if isinstance(select_id, (list, np.ndarray)):
                        idxs = [int(i) for i in np.array(select_id).flatten()]
                    else:
                        idxs = [int(select_id)]

                    del_list.extend(idxs)  # 保存删除的索引

                    # 删除对应框、文字和分数
                    bboxes_copy = np.delete(np.array(bboxes_copy), idxs, axis=0)
                    txts = np.delete(np.array(txts), idxs, axis=0).tolist()
                    scores = np.delete(np.array(scores), idxs, axis=0).tolist()
                    boxes = np.delete(np.array(boxes), idxs, axis=0).tolist()

                    # 重新绘制图像
                    img_bbox, _, _ = draw_ocr(img.copy(), boxes, txts, scores, font_path=font_path)
                    img_render_copy = img_bbox.copy()

                    # 重置标志
                    select_bbox_flag = False
                    select_all_bbox_flag = False
                    select_id = None

            # vis mesh and original img
            elif keyboardSet == ord('s') or keyboardSet == ord('S'):
                success_save = True
                break


            elif keyboardSet == ord('a') or keyboardSet == ord('A') or add_box:
                # wait for bbox
                add_box = True
                if selected_a_bbox_flag:
                    user_input = input("Enter text: ")

                    # update text
                    txts.append(user_input)
                    scores.append(1.0)
                    bboxes_copy = np.concatenate((bboxes_copy, np.array(added_box).reshape(1,2,2)), axis=0)
                    poly = [[added_box[0],added_box[1]],[added_box[2],added_box[1]],[added_box[2],added_box[3]],[added_box[0],added_box[3]]]
                    boxes.append(poly)

                    img_bbox, _, _ = draw_ocr(img.copy(), boxes, txts, scores, font_path=font_path)
                    img_render_copy = img_bbox.copy()

                    selected_a_bbox_flag = False
                    add_box = False
                    added_box = [0,0,0,0]

            # next image
            elif keyboardSet == 32:
                break

            if show_flag == 0 and all_render_flag == 0 and vis_output_flag == 0:
                show_img = show_img_selected
            elif show_flag == 1 and all_render_flag == 0 and vis_output_flag == 0:
                show_img = img_render_copy
            elif show_flag == 2 and all_render_flag == 0 and vis_output_flag == 0:
                show_img = img
            # elif vis_output_flag == 1 and show_flag == 0 and all_render_flag == 0:
            #     show_img = selected_param_render
            # elif all_render_flag == 1 and show_flag == 0 and vis_output_flag == 0:
            #     show_img = ori_img_render
            else:
                show_img = show_img_selected
                show_flag = 0
                all_render_flag = 0
                vis_output_flag = 0

            show_img_wh = np.array([show_img.shape[1], show_img.shape[0]])
            if scale != 1.0 and (show_img_wh != img_patch_wh).all():
                show_img = show_img[win_location[1]:win_location[1] + img_patch_wh[1], win_location[0]:win_location[0] + img_patch_wh[0]]

            if show_img.shape[0] == 0:
                vis_img_ = img_render_copy.copy()
                vis_img_[win_location_bbox[1]:win_location_bbox[1] + img_patch_wh_bbox[1], win_location_bbox[0]:win_location_bbox[0] + img_patch_wh_bbox[0]] = selected_bbox_img
                show_img = vis_img_[win_location[1]:win_location[1] + img_patch_wh[1], win_location[0]:win_location[0] + img_patch_wh[0]]

            cv2.imshow("image", show_img)

            txt_img = update_text_vis(img, txts, scores, select_id, font_path=font_path)

            cv2.imshow("text", txt_img)
            keyboardSet = cv2.waitKey(1)

    return boxes, txts, scores, img_bbox, success_save


    