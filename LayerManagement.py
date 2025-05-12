import cv2
from PyQt6.QtGui import QImage, QPixmap
import ProcessingFunctions as PF
import numpy as np


def rename_layer(layer_name, cur_id):
    if cur_id==0:
        return layer_name
    else:
        return f"{layer_name}_{cur_id}"


class LayerManagement:
    def __init__(self):
        self.scaled_size = None
        self.names = []
        self.imgs = []      # cv2.Mat
        self.state = []     # bool: True for displayed

    # params:
    # img: must be a RGB cv2.Mat
    def set_original_img(self, img_name, img):
        self.names = [img_name]
        self.imgs = [img]
        self.state = [True]
        self.scaled_size = img.shape[:-1]

    def update_original_img(self, img):
        if self.imgs:
            self.imgs[0] = img
            self.scaled_size = img.shape[:-1]
        else:
            print('Nothing to update!')


    # params:
    # img can be a 3-channel cv2.Mat
    #              or a binary mask layer of 0/1 matrix and a layer_color
    # ret: the true layer_name that is unique
    def add_layer(self, layer_name, img, is_contour=False):
        if not self.names:
            self.set_original_img(layer_name, img)
            return layer_name
        cur_id = 0
        for _layer_name in self.names:
            if _layer_name == rename_layer(layer_name, cur_id):
                cur_id += 1
        layer_name = rename_layer(layer_name, cur_id)
        print(f"img[0].shape:{img[0].shape}")
        if is_contour:
            # deprecated because we need resize the figure now
            # tmp, _, _ = PF.obtain_contours(img[0], img[1], 2)
            # self.imgs.append((tmp, img[1], img[2]))
            self.imgs.append((img[0] , img[1], img[2]))
        else:
            self.imgs.append(img)
        self.names.append(layer_name)
        self.state.append(True)
        return layer_name

    # @ret: True if can delete, False if cannot
    def remove_layer(self, layer_name):
        for index, _name in enumerate(self.names):
            if _name == layer_name:
                if index == 0:
                    print("Cannot delete original image. Exit current file to delete it.")
                    return False
                else:
                    del self.names[index]
                    del self.imgs[index]
                    del self.state[index]
                    return True
        print(f"No such layer:{layer_name}.")
        return False

    # func: Can modify current state of all layers, including the original image
    # ret: If modification makes a difference, return True,
    #      else (current state = new state, or layer not found return False.
    def modify_layer_state(self, layer_name, new_state):
        for index, _name in enumerate(self.names):
            if _name == layer_name:
                if self.state[index] != new_state:
                    self.state[index] = new_state
                    return True
                else:
                    return False
        return False

    # ret: (in the order from top to bottom)
    # list_1: layer_names (str) of active layers
    # list_2: imgs (cv2.Mat) of active layers
    def get_active_layers(self):
        list_1 = []
        list_2 = []
        for index, state in enumerate(self.state):
            if state:
                list_1.append(self.names[index])
                list_2.append(self.imgs[index])
        return list_1, list_2

    def get_active_img(self):
        '''
        get the last viewable image layer
        :return: img or None
        '''
        for index in range(len(self.state)-1, -1, -1):
            if self.state[index] and not isinstance(self.imgs[index], tuple):
                return cv2.resize(self.imgs[index], self.scaled_size)
        return None

    def get_layer_by_name(self, layername):
        """

        :param layername: the name of target layer
        :return: resized img, type indicator string
        """
        try:
            i = self.names.index(layername)
            if isinstance(self.imgs[i], tuple):
                return cv2.resize(self.imgs[i][0], (self.scaled_size[1], self.scaled_size[0])), 'Mask'
            else:
                return cv2.resize(self.imgs[i], (self.scaled_size[1], self.scaled_size[0])), 'Image'
        except ValueError:
            print(f'No such layer: {layername}.')
            return None

    def get_layer_type(self, layername):
        try:
            i = self.names.index(layername)
            if isinstance(self.imgs[i], tuple):
                return 'Mask'
            else:
                return 'Image'
        except ValueError:
            print(f'No such layer: {layername}.')
            return None


    def get_mixed_img(self, mask_flag=True, contour_flag=True):
        print(f"layer management, size:{self.scaled_size}")
        if self.imgs:
            print(f"imgs[0].shape: {self.imgs[0].shape}")
        names, imgs = self.get_active_layers()
        if len(names) == 0:     # no images
            if self.imgs:
                return np.zeros(self.scaled_size, np.uint8)
            else:
                return None
        else:                   # have images
            ret_img = np.zeros(self.scaled_size, np.uint8)
            _mask = np.zeros(self.scaled_size, np.uint8)
            has_mask_flag = False
            _contours = []
            for img in imgs:
                if isinstance(img, tuple):  # mask
                    if img[2]=='Mask':      # currently, masks stacks are treated as taking union
                        cur_img = cv2.resize(img[0], (self.scaled_size[1], self.scaled_size[0]))
                        print(f"_mask.shape:{_mask.shape}, img[0].shape:{img[0].shape}, _mask.max:{_mask.max()}, img[0].max:{img[0].max()}")
                        print(f"cur_img.sha[e:{cur_img.shape}, cur_img.max:{cur_img.max()}, self.scaled_size:{self.scaled_size}")
                        # cv2.imshow('mask_de', cur_img)
                        _mask = cv2.bitwise_or(_mask, cur_img)
                        has_mask_flag = True
                    elif img[2]=='Contour':
                        cur_img = cv2.resize(img[0], (self.scaled_size[1], self.scaled_size[0]))
                        tmp, _, _ = PF.obtain_contours(cur_img, img[1], 2)
                        _contours.append((tmp, img[1], 2))
                else:   # RGB image / grayscale img
                    if len(img.shape) == 3:   # RGB image
                        # ret_img = np.copy(img)
                        ret_img = cv2.resize(img, (self.scaled_size[1], self.scaled_size[0]))
                    else:                   # grayscale image
                        ret_img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (self.scaled_size[1], self.scaled_size[0]))
        if len(ret_img.shape)==3:
            ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)
        elif len(ret_img.shape)==2:
            ret_img = cv2.cvtColor(ret_img, cv2.COLOR_GRAY2RGB)
        if mask_flag and has_mask_flag:
            ret_img *= cv2.cvtColor(_mask, cv2.COLOR_GRAY2BGR)  # apply mask
        if contour_flag:
            for _contour, color, thickness in _contours:
                print(f'reach here')
                r_thickness = int((self.scaled_size[0]+self.scaled_size[1])/1000)+1
                cv2.drawContours(ret_img, _contour, -1, color, r_thickness)
        return ret_img

    def move_up(self, layer_name):
        for index, name in enumerate(self.names):
            if name==layer_name:
                if index!=0:
                    self.names[index - 1], self.names[index] = self.names[index], self.names[index - 1]
                    self.imgs[index - 1], self.imgs[index] = self.imgs[index], self.imgs[index - 1]
                    self.state[index - 1], self.state[index] = self.state[index], self.state[index - 1]
                    return True
                else:
                    return False
        return False

    def move_down(self, layer_name):
        for index, name in enumerate(self.names):
            if name==layer_name:
                if index!=len(self.names)-1:
                    self.names[index + 1], self.names[index] = self.names[index], self.names[index + 1]
                    self.imgs[index + 1], self.imgs[index] = self.imgs[index], self.imgs[index + 1]
                    self.state[index + 1], self.state[index] = self.state[index], self.state[index + 1]
                    return True
                else:
                    return False
        return False


if __name__=="__main__":
    LM = LayerManagement()
    img = cv2.imread("./process_test/images/img.png")
    # grayscale = PF.process_grayscale(img)
    mean = PF.process_filter(img, "Mean Filter", {'ksize':(5,5)})
    mask = PF.obtain_mask(img)
    LM.set_original_img("img", img)
    LM.add_layer("mean", mean)
    LM.add_layer("mask", (mask, (255, 0, 0)))
    res = LM.get_mixed_img()
    cv2.imshow("res", res)
    cv2.waitKey(0)

    LM.modify_layer_state("mean", False)
    res = LM.get_mixed_img()
    cv2.imshow("res", res)
    cv2.waitKey(0)

    LM.modify_layer_state("img", False)
    res = LM.get_mixed_img()
    cv2.imshow("res", res)
    cv2.waitKey(0)

    LM.modify_layer_state("mask", False)
    res = LM.get_mixed_img()
    cv2.imshow("res", res)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
