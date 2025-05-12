import cv2
import numpy as np
import skimage.measure
from skimage.segmentation import flood_fill
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes, generic_filter
from collections import deque
from tqdm import tqdm, trange


def process_FNL(img):
    value = 2
    dst = cv2.fastNlMeansDenoisingColored(src=img, dst=None, h=value, hColor=value,
                                          templateWindowSize=7,
                                          searchWindowSize=21)
    return dst


MEAN_FILTER = 0
GAUSSIAN_FILTER = 1
MEDIAN_FILTER = 2
GRAY_SCALE = 3
BINARY = 4
ERODE = 5
DILATION = 6
OPEN = 7
CLOSE = 8


# filter_params = {ksize : kernel size, always tuple,
#                  sigma: param for GaussianBlur}
def process(img, process_type, params : dict):
    if process_type == MEAN_FILTER:
        return cv2.blur(img, params['ksize'])       # ksize : tuple
    elif process_type == GAUSSIAN_FILTER:
        ksize = [params['ksize'][0], params['ksize'][1]]
        if ksize[0]%2==0:
            ksize[0] += 1
        if ksize[1]%2==0:
            ksize[1] += 1
        print(f'resize ksize:{ksize}')
        return cv2.GaussianBlur(img, ksize, params['sigma'])      # ksize : tuple
    elif process_type == MEDIAN_FILTER:
        return cv2.medianBlur(img, params['ksize'][0])     # ksize : (int,)
    elif process_type == GRAY_SCALE:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif process_type == BINARY:
        return process_binary(img)
    elif process_type == ERODE:
        return process_erode(img, params['ksize'])
    elif process_type == DILATION:
        return process_dilation(img, params['ksize'])
    elif process_type == OPEN:
        return process_open(img, params['ksize'])
    elif process_type == CLOSE:
        return process_close(img, params['ksize'])
    elif process_type is None:
        return img
    else:
        raise ValueError(f"Not valid process type: {process_type}")


def process_grayscale(img):
    dst = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return dst


def process_binary(img):
    if len(img.shape) != 2: # colored image
        img = process_grayscale(img)
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def process_erode(img, kernel_size=(5,5)):
    kernel = np.ones(kernel_size, np.uint8)
    dst = cv2.erode(img, kernel, iterations=1)
    return dst


def process_dilation(img, kernel_size=(5,5)):
    kernel = np.ones(kernel_size, np.uint8)
    dst = cv2.dilate(img, kernel, iterations=1)
    return dst


def process_open(img, kernel_size=(5,5)):
    kernel = np.ones(kernel_size, np.uint8)
    dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return dst


def process_close(img, kernel_size=(5,5)):
    kernel = np.ones(kernel_size, np.uint8)
    dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return dst


# ret: a mask matrix made up of 0 / 1. Multiply by 255 to obtain a viewable image.
def obtain_mask_grabCut(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, 290, 570)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2


def obtain_contours_grabCut(img):
    mask2 = obtain_mask_grabCut(img)*255
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    black_img = np.zeros(img.shape, np.uint8)
    cv2.drawContours(black_img, contours, -1, (0, 255, 0), 2)
    return black_img


def draw_contours(img, mask, color, thickness):
    ret_img = img
    mask2 = mask*255
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ret_img, contours, -1, color, thickness)
    return ret_img


def obtain_contours(mask, color, thickness):
    mask2 = mask*255
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours, color, thickness


FOREGROUND_GRABCUT = 0
FOREGROUND_HSV = 1
ROI_HSV = 2


# params:
# ksize-kernel size
def obtain_foreground_mask(img, operation_type, remove_noise_flag=True):
    if operation_type==FOREGROUND_GRABCUT:
        mask = obtain_mask_grabCut(img)
    elif operation_type==FOREGROUND_HSV:
        mask = obtain_mask_hsv(img, 'fore')
    elif operation_type==ROI_HSV:
        mask = obtain_mask_hsv(img, 'roi')
    else:
        mask = None
    if remove_noise_flag:
        mask = remove_small_noise(mask)
    return mask


def obtain_mask_hsv(img : cv2.Mat, mask_type):
    dst = color_correction(img)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    if mask_type=='fore':
        lower_bound = np.array([80, 40, 40], dtype=np.uint8)
        upper_bound = np.array([180, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(dst, lower_bound, upper_bound)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def remove_small_noise(mask, area_lowerb=100):
    # 2. fill in small holes inside the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(mask, [cnt], 0, 1, -1)  # -1 表示填充

    # 1. remove small dots outside the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_lowerb:
            cleaned_mask[labels == i] = 1

    return cleaned_mask


def color_correction(img, mask=None):
    # return rgb_correction(img)
    res = hsv2_correction(img)
    if mask is not None:
        res = (res * cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) + img * cv2.cvtColor(1-mask, cv2.COLOR_GRAY2BGR))
    return res


def rgb_correction(img):
    b, g, r = cv2.split(img)
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)

    max_avg = max(avg_b, avg_g, avg_r)
    scale_b = max_avg / avg_b
    scale_g = max_avg / avg_g
    scale_r = max_avg / avg_r
    b = cv2.convertScaleAbs(b, alpha=scale_b)
    g = cv2.convertScaleAbs(g, alpha=scale_g)
    r = cv2.convertScaleAbs(r, alpha=scale_r)

    corrected_img = cv2.merge((b, g, r))
    return corrected_img


def hsv_correction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    equ2 = cv2.equalizeHist(v)
    equ1 = cv2.equalizeHist(s)
    new_hsv = cv2.merge((h, equ1, equ2))
    corrected_img = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)
    # corrected_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return corrected_img


def hsv2_correction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    v_eq = clahe.apply(v)
    s_eq = clahe.apply(s)
    new_hsv = cv2.merge((h, s_eq, v_eq))
    corrected_img = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
    return corrected_img


def segment_regions(mask, remove_small_flag=False):
    '''
    segment the mask

    :param mask: a binary mask
    :return: a list of segmented masks
    '''
    labeled_mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(labeled_mask)

    threshold_area = int(mask.sum()/len(regions)*0.2)

    segmented_masks = []
    for region in regions:
        if remove_small_flag and region.area < threshold_area:
            print(f'skip one area of size {region.area}')
            continue
        region_mask = np.zeros_like(mask, np.uint8)
        region_mask[labeled_mask == region.label] = 1
        segmented_masks.append(region_mask)

    return segmented_masks


# todo: fix strange shape
def save_ROI(img, mask, output_path):
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    if mask.max() == 1:
        mask = mask * 255
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)

    roi_img = img[y:y + h, x:x + w].copy()
    roi_mask = mask[y:y + h, x:x + w].copy()

    bgra = cv2.cvtColor(roi_img, cv2.COLOR_BGR2BGRA)

    bgra[:, :, 3] = np.where(roi_mask == 255, 255, 0)
    cv2.imwrite(output_path, bgra)


def mean_color_value(img, mask):
    img = img * cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return img.sum()/mask.sum()


def obtain_ROI_mask(img, ksize_ratio=0.01, remove_noise_flag=True):
    """
    detect the blue dyed areas in the image.
    Possible Solutions:
    1. Gaussian Filter (optional)
    2. Color Correction * 1~2 times
    3. HSV inRange detect
    4. remove noises
    5. regional grow
    :param img: the image to process
    :param params: scale of kernel for noise remove
    :param remove_noise_flag: whether remove noises
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    foreground_mask = obtain_foreground_mask(img, FOREGROUND_HSV)

    ###1 Gaussian Filter
    params = {'ksize':(int(img.shape[0]*ksize_ratio), int(img.shape[1]*ksize_ratio)), 'sigma':1.0}
    dst = process(img, GAUSSIAN_FILTER, params)
    print(f'img_value:{img.sum()}, dst_value:{dst.sum()}')
    # dst = img.copy()

    ###2 Color Correction
    dst = color_correction(dst, mask=foreground_mask)
    color_upper_threshold = 560
    # color_lower_threshold = 350
    msv = mean_color_value(dst, foreground_mask)
    # while msv>color_upper_threshold or msv<color_lower_threshold:
    while msv>color_upper_threshold:
        dst = color_correction(dst, mask=foreground_mask)
        msv = mean_color_value(dst, foreground_mask)
        print(f'color correction after: {msv}')

    ###3 HSV inRange
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([140, 20, 60], dtype=np.uint8)
    upper_bound = np.array([250, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    ###4 foreground constraint
    # foreground_mask = obtain_foreground_mask(img, operation_type='fore')
    # print(f'mask_shape:{mask.shape}, foreground_mask_shape:{foreground_mask.shape}')
    # mask = cv2.bitwise_xor(mask, foreground_mask, mask=foreground_mask)
    # mask = cv2.bitwise_xor(mask, foreground_mask)
    # mask = cv2.bitwise_and(mask, foreground_mask)
    # mask = (255-mask) * foreground_mask
    # print(f'mask_val:{mask.max()}, fore_mask_val:{foreground_mask.max()}')

    ###5 remove noise
    kernel = np.ones(params['ksize'], np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if remove_noise_flag:
        mask = remove_small_noise(mask, area_lowerb=200)

    ###6 Regional Grow
    mask = region_growing(hsv, mask, color_threshold=10, distance_threshold=1)

    return mask


def region_growing(hsv, initial_mask, color_threshold=10, distance_threshold=1):
    """
    区域生长算法实现

    参数:
        image: 原始BGR图像
        initial_mask: 初始mask(0和1的数组)
        color_threshold: 颜色相似性阈值
        distance_threshold: 生长距离阈值(像素)

    返回:
        生长后的mask
    """
    # 将图像转换为HSV颜色空间
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 获取初始mask中的种子点
    seed_points = np.argwhere(initial_mask > 0)

    # 如果没有种子点，直接返回空mask
    if len(seed_points) == 0:
        return np.zeros_like(initial_mask)

    # 计算种子区域的平均颜色(HSV)
    mean_hue = np.mean(hsv[initial_mask > 0][:, 0])
    mean_sat = np.mean(hsv[initial_mask > 0][:, 1])
    mean_val = np.mean(hsv[initial_mask > 0][:, 2])

    # 创建生长后的mask
    grown_mask = initial_mask.copy()

    # 定义8邻域
    neighbors = []
    for x in range(-distance_threshold,distance_threshold+1):
        for y in range(-distance_threshold, distance_threshold+1):
            if x==y==0:
                continue
            neighbors.append((x,y))

    # 使用队列实现区域生长
    queue = deque(seed_points.tolist())
    visited = set((x, y) for x, y in seed_points)

    while queue:
        x, y = queue.popleft()

        # 检查8邻域
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy

            # 检查边界
            if (nx < 0 or ny < 0 or nx >= hsv.shape[0] or ny >= hsv.shape[1]):
                continue

            # 如果已经访问过，跳过
            if (nx, ny) in visited:
                continue

            visited.add((nx, ny))

            # 获取当前像素的HSV值
            h, s, v = hsv[nx, ny]

            # 计算颜色距离(考虑色调的循环特性)
            hue_diff = min(abs(h - mean_hue), 180 - abs(h - mean_hue))
            sat_diff = abs(s - mean_sat)
            val_diff = abs(v - mean_val)

            # 综合颜色差异(可以调整权重)
            color_diff = 0.5 * hue_diff + 0.3 * sat_diff + 0.2 * val_diff

            # 如果颜色相似，加入生长区域
            if color_diff < color_threshold:
                grown_mask[nx, ny] = 1
                queue.append((nx, ny))

    # 使用形态学操作填充小孔洞和连接断片
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # grown_mask = cv2.morphologyEx(grown_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # 可选: 使用flood fill填充完全封闭的区域
    # 这里使用更高效的skimage实现
    grown_mask = flood_fill(grown_mask, (0, 0), 2)
    grown_mask = np.where(grown_mask == 2, 0, grown_mask)

    return grown_mask


def auto_color_correction(img):
    foreground_mask = obtain_foreground_mask(img, FOREGROUND_HSV)
    color_threshold = 600
    msv = mean_color_value(img, foreground_mask)
    while msv > color_threshold:
        img = color_correction(img)
        msv = mean_color_value(img, foreground_mask)
        print(f'color correction after: {msv}')
    return img


def obtain_ROI_mask2(img, threshold_level = 1.1):
    fore_mask = obtain_foreground_mask(img, FOREGROUND_HSV)
    b, g, r = cv2.split(img)
    mean_br_ratio = cv2.bitwise_and(b, fore_mask).sum() / cv2.bitwise_and(r, fore_mask).sum()
    threshold_br_ratio = mean_br_ratio * threshold_level
    print('threshold_br_ratio:{}'.format(threshold_br_ratio))
    res_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if fore_mask[x][y]>0:
                br_ratio = b[x][y]/r[x][y]
                res_mask[x][y] = br_ratio * 255
                # if br_ratio>threshold_br_ratio:
                #     res_mask[x][y] = 1
                #     print(br_ratio, end=', ')
    return res_mask


def convex_contour(roi_mask):
    """
    Draw convexes or rectangles to show the roi regions
    :param roi_mask: a single 0/1 roi_mask
    :return: a contour mask
    """
    labeled_mask = skimage.measure.label(roi_mask)
    regions = skimage.measure.regionprops(labeled_mask)

    contoured_mask = np.zeros_like(roi_mask, dtype=np.uint8)

    for region in regions:
        coords = region.coords
        hull = cv2.convexHull(coords[:, ::-1])
        cv2.fillPoly(contoured_mask, [hull], color=1)

    return contoured_mask


def smooth_expand_mask(mask, expand_ratio=0.01, smooth_kernel=(5, 5)):
    """
    通过膨胀 + 高斯滤波实现轮廓平滑扩展
    :param mask: 输入二值掩膜 (0/1)
    :param expand_ratio: the ratio of expanding radius over mask size
    :param smooth_kernel: 高斯滤波核大小（控制平滑度）
    :return: 平滑扩展后的 0/1 掩膜
    """
    # 1. 转换为 uint8 格式（0/255）
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 2. 膨胀操作扩展轮廓
    r = int(min(mask.shape)*expand_ratio)*2+1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    expanded_mask = cv2.dilate(mask_uint8, kernel)

    # 3. 高斯滤波平滑边缘
    smoothed_mask = cv2.GaussianBlur(expanded_mask, smooth_kernel, 0)

    # 4. 重新二值化（闭合轮廓）
    _, result_mask = cv2.threshold(smoothed_mask, 1, 1, cv2.THRESH_BINARY)
    # print(f'result_mask:{result_mask}')

    return result_mask


def fast_local_std(gray: np.ndarray, size: int = 15) -> np.ndarray:
    """Fast local standard deviation using convolution."""
    gray = gray.astype(np.float32)
    mean = cv2.blur(gray, (size, size))
    mean_sq = cv2.blur(gray ** 2, (size, size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    return std


def detect_portal_areas_texture(image: np.ndarray):
    """
    Detect portal areas using only grayscale and texture features (no color).

    Parameters:
        image: RGB image (used only to convert to grayscale)
        mask: Binary mask (foreground = 1)

    Returns:
        cleaned_portal_mask: mask of portal areas
        normal mask: mask of normal portal areas
        fibre mask: mask of fibred portal areas
        info: information for report
    """
    h = image.shape[0]
    w = image.shape[1]
    lower_img = cv2.resize(image, (int(w/10), int(h/10)))
    fore_mask = obtain_foreground_mask(lower_img, FOREGROUND_HSV)
    fore_mask = cv2.resize(fore_mask, (w, h))

    # 1. Convert to grayscale and apply mask
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=fore_mask.astype(np.uint8))

    # 2. Compute local texture map
    texture = fast_local_std(gray, size=7)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Threshold the texture map (Otsu)
    thresh = skimage.filters.threshold_otsu(texture)
    binary = (texture < thresh).astype(np.uint8)
    binary = cv2.bitwise_and(binary, binary, mask=fore_mask.astype(np.uint8))

    # 4. Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    # 闭运算：合并相邻区域
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(closed, [cnt], 0, 1, -1)  # -1 表示填充
    portal_mask = cv2.bitwise_and((1 - closed), fore_mask.astype(np.uint8))

    portal_mask = cv2.erode(portal_mask, kernel)
    kernel = np.ones((80, 80), dtype=np.uint8)
    portal_mask = cv2.dilate(portal_mask, kernel)

    r1, r2, r3, r4 = classify_portal_regions(portal_mask, texture, fore_mask)

    return r1, r2, r3, r4


def classify_portal_regions(portal_mask: np.ndarray, texture_map: np.ndarray, fore_mask : np.ndarray):
    """
    too long function so I segment
    """
    # 连通区域标记
    labeled = skimage.measure.label(portal_mask)

    # 计算区域属性（带纹理信息）
    props = skimage.measure.regionprops(labeled, intensity_image=texture_map)

    normal_portals = []
    fibrotic_portals = []

    cleaned_portal_mask = np.zeros_like(portal_mask)

    for region in props:
        area = region.area
        mean_texture = region.mean_intensity
        solidity = region.solidity
        eccentricity = region.eccentricity
        cur_label = region.label
        if area<10000:
            continue

        cleaned_portal_mask[labeled == cur_label] = 1

        # 简单规则（可调）：面积大、纹理高、形状不规则 --> 纤维化
        if area > 3000 and mean_texture > 25 and (solidity < 0.8 or eccentricity > 0.8):
            fibrotic_portals.append(region)
        else:
            normal_portals.append(region)

    normal_mask = np.zeros_like(portal_mask)
    fibre_mask = np.zeros_like(portal_mask)
    info = {'portal_num': len(props),
            'normal_portal_num': len(normal_portals),
            'fibre_portal_num': len(fibrotic_portals),
            'portal_area': None,
            'normal_portal_area': 0,
            'fibre_portal_area': 0
            }
    tmp = 0
    for region in normal_portals:
        coords = region.coords
        tmp += region.area
        for y, x in coords:
            normal_mask[y, x] = 1
    info['normal_portal_area'] = tmp / fore_mask.sum()

    tmp = 0
    for region in fibrotic_portals:
        tmp += region.area
        coords = region.coords
        for y, x in coords:
            fibre_mask[y, x] = 1
    info['fibre_portal_area'] = tmp / fore_mask.sum()
    info['portal_area'] = info['fibre_portal_area'] + info['normal_portal_area']

    return cleaned_portal_mask, normal_mask, fibre_mask, info


def compute_mean_texture_per_region(labeled, texture_map):
    mean_texture = np.zeros(np.max(labeled) + 1)
    counts = np.zeros_like(mean_texture)

    print(f'reached')

    for y in trange(labeled.shape[0]):
        for x in range(labeled.shape[1]):
            label_val = labeled[y, x]
            if label_val == 0:
                continue
            mean_texture[label_val] += texture_map[y, x]
            counts[label_val] += 1

    mean_texture = np.divide(mean_texture, counts, out=np.zeros_like(mean_texture), where=counts > 0)
    return mean_texture


# def local_std_filter(gray, size=15):
#     """Compute local standard deviation (as texture indicator)."""
#     def std_func(values):
#         return np.std(values)
#     return generic_filter(gray.astype(float), std_func, size=(size, size))




# todo:种子像素生长修正
# todo:判断color correction次数
