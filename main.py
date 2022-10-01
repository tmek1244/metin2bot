import cv2
import numpy as np
from mss import mss
import time


# COLORS IN HSV
metins_colors = [
    # [(55, 187, 85), (57, 168, 80)], # LVL 5
    [(75, 75, 165), (110, 155, 255)], # LVL 10
    # [(28, 145, 96), (40, 255, 255)], # LVL 15
    # [(113, 113, 54), (240, 240, 160)],
    # [(113, 49, 74), (255, 158, 208)]
    # [(150, 50, 50), (180, 255, 255)] # LVL 20
]

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return [x, y, w, h]

def _intersect(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if h<0 or w<0:                                              # in original code :  if w<0 or h<0:
        return False
    return True

def close(a,b):
    a_X, a_Y = (a[0]+a[2])/2, (a[1]+a[3])/2
    b_X, b_Y = (b[0]+b[2])/2, (b[1]+b[3])/2
    dist = np.sqrt(np.power(a_X - b_X, 2)+np.power(a_Y- b_Y, 2))
    return dist < min((a[2]+a[3]+b[2]+b[3])/2, 20)

def _group_rectangles(rec):
    """
    Uion intersecting rectangles.
    Args:
        rec - list of rectangles in form [x, y, w, h]
    Return:
        list of grouped ractangles 
    """
    tested = [False for i in range(len(rec))]
    final = []
    i = 0
    while i < len(rec):
        if not tested[i]:
            j = i+1
            while j < len(rec):
                if not tested[j] and (close(rec[i], rec[j]) or _intersect(rec[i], rec[j])):
                    rec[i] = union(rec[i], rec[j])
                    tested[j] = True
                    j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Display', 1920, 1080)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mark_objects(input, colors):
    image = cv2.blur(input, (4, 4))
    contrast_mask = cv2.cvtColor(apply_brightness_contrast(input, 0, 120), cv2.COLOR_RGB2GRAY)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # viewImage(hsv_img)
    # color_0 = cv2.cvtColor(np.uint8([[colors[0]]]),cv2.COLOR_RGB2HSV)
    # color_1 = cv2.cvtColor(np.uint8([[colors[1]]]),cv2.COLOR_RGB2HSV)
    # print(colors[0], colors[1])
    # contrast_mask = cv2.cvtColor(apply_brightness_contrast(hsv_img, 0, 127), cv2.COLOR_BGR2GRAY)
    curr_mask = cv2.inRange(hsv_img, colors[0], colors[1])
    hsv_img[contrast_mask > 0] = (255, 255, 255)
    hsv_img[contrast_mask == 0] = (0, 0, 0)
    # hsv_img[curr_mask > 0] = (255, 255, 255)
    hsv_img[curr_mask == 0] = (0, 0, 0)
    RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray, 90, 255, 0)
    contours, _ =  cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    viewImage(contrast_mask)
    viewImage(curr_mask)
    areas = sorted(contours, key=cv2.contourArea, reverse=True)
    filtered_contours = areas[:25]
    rectangles = [cv2.boundingRect(c) for c in filtered_contours]
    rectangles = _group_rectangles(rectangles)
    # print(rectangle)
    for rectangle in rectangles:
        x,y,w,h = rectangle
        # if 0.5 < w/h < 1.5:
        cv2.rectangle(input,(x,y),(x+w,y+h),(0,255,0),1)


def mark_metins(image, metins=metins_colors):
    # test = cv2.imread(path)
    for metin_color in metins:
        mark_objects(image, metin_color)

    # viewImage(image)
    return image


def capture_n_mark():
    bounding_box = {'top': 100, 'left': 0, 'width': 1700, 'height': 900}

    sct = mss()

    while True:
        last_time = time.time()
        sct_img = sct.grab(bounding_box)
        image = mark_metins(np.array(sct_img))
        cv2.imshow('screen', image)
        # cv2.imshow('screen', np.array(sct_img))

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break

# Function to perform Difference of Gaussians
def difference_of_Gaussians(img, k1, s1, k2, s2):
    b1 = cv2.GaussianBlur(img,(k1, k1), s1)
    b2 = cv2.GaussianBlur(img,(k2, k2), s2)
    return b1 - b2


def main():   
    for i in range(3, 4):
        image = cv2.imread(f'images/{i}.png')
        resized = cv2.resize(image, (1920, 1080))
        # viewImage(resized)
        mark_metins(resized)
        # resized = apply_brightness_contrast(resized, 0, 120)
        viewImage(resized)

    # resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # for i in range(3, 10, 2):
    #     for j in range(3, 10, 2):
    #         for k in range(3, 10, 2):
    #             for l in range(3, 10, 2):
    #                 DoG_img = difference_of_Gaussians(resized_gray, i, j, k, l)

    # resized = cv2.blur(resized, (10, 10))
    # mark_metins(resized, [metins_colors[0]])
    # # mark_metins(f'images/{i}.png')
    # viewImage(resized)
    # for i in range(11, 12):
    #     test = cv2.imread(f'images/{i}.png')
    #     resized = cv2.resize(test, (1920, 1080))
    #     resized = cv2.blur(resized, (10, 10))
    #     mark_metins(resized, [metins_colors[0]])
    #     # mark_metins(f'images/{i}.png')
    #     viewImage(resized)

            
        # print("fps: {}".format(1 / (time.time() - last_time)))


if __name__ == '__main__':
    main()
