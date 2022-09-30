import cv2
import numpy as np
from mss import mss
import time


# COLORS IN HSV
metins_colors = [
    # [(55, 187, 85), (57, 168, 80)], # LVL 5
    [(84, 96, 200), (105, 150, 255)], # LVL 10
    [(28, 145, 96), (40, 255, 255)], # LVL 15
    # [(113, 113, 54), (240, 240, 160)],
    # [(113, 49, 74), (255, 158, 208)]
    [(150, 50, 50), (180, 255, 255)] # LVL 20
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


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mark_objects(test, colors):
    hsv_img = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)
    # viewImage(hsv_img)
    # color_0 = cv2.cvtColor(np.uint8([[colors[0]]]),cv2.COLOR_RGB2HSV)
    # color_1 = cv2.cvtColor(np.uint8([[colors[1]]]),cv2.COLOR_RGB2HSV)
    # print(colors[0], colors[1])
    curr_mask = cv2.inRange(hsv_img, colors[0], colors[1])
    # viewImage(curr_mask)
    hsv_img[curr_mask > 0] = (255, 255, 255)
    hsv_img[curr_mask == 0] = (0, 0, 0)
    RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray, 90, 255, 0)
    contours, _ =  cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # viewImage(hsv_img)
    areas = sorted(contours, key=cv2.contourArea, reverse=True)
    filtered_contours = areas[:25]
    rectangles = [cv2.boundingRect(c) for c in filtered_contours]
    rectangles = _group_rectangles(rectangles)
    # print(rectangles)
    for rectangle in rectangles:
        x,y,w,h = rectangle
        if w*h > 300:
            cv2.rectangle(test,(x,y),(x+w,y+h),(0,255,0),1)


def mark_metins(image):
    # test = cv2.imread(path)
    for metin_color in metins_colors:
        mark_objects(image, metin_color)

    # viewImage(image)
    cv2.imshow('screen', image)


def main():
    # for i in range(1, 12):
    #     mark_metins(f'images/{i}.png')
    bounding_box = {'top': 100, 'left': 0, 'width': 1700, 'height': 900}

    sct = mss()

    while True:
        last_time = time.time()
        sct_img = sct.grab(bounding_box)
        mark_metins(np.array(sct_img))
        # cv2.imshow('screen', np.array(sct_img))

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
            
        # print("fps: {}".format(1 / (time.time() - last_time)))


if __name__ == '__main__':
    main()
