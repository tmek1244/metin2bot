import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random as rng


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
    cv.namedWindow('Display', cv.WINDOW_NORMAL)
    cv.imshow('Display', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def mark_objects(test, colors):
    hsv_img = cv.cvtColor(test, cv.COLOR_BGR2HSV)
    # viewImage(hsv_img)
    # color_0 = cv.cvtColor(np.uint8([[colors[0]]]),cv.COLOR_RGB2HSV)
    # color_1 = cv.cvtColor(np.uint8([[colors[1]]]),cv.COLOR_RGB2HSV)
    # print(colors[0], colors[1])
    curr_mask = cv.inRange(hsv_img, colors[0], colors[1])
    # viewImage(curr_mask)
    hsv_img[curr_mask > 0] = (255, 255, 255)
    hsv_img[curr_mask == 0] = (0, 0, 0)
    RGB_again = cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)
    gray = cv.cvtColor(RGB_again, cv.COLOR_RGB2GRAY)
    _, threshold = cv.threshold(gray, 90, 255, 0)
    contours, _ =  cv.findContours(threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    # viewImage(hsv_img)
    areas = sorted(contours, key=cv.contourArea, reverse=True)
    filtered_contours = areas[:25]
    rectangles = [cv.boundingRect(c) for c in filtered_contours]
    rectangles = _group_rectangles(rectangles)
    # print(rectangles)
    for rectangle in rectangles:
        x,y,w,h = rectangle
        if w*h > 300:
            cv.rectangle(test,(x,y),(x+w,y+h),(0,255,0),1)


def mark_metins(path):
    test = cv.imread(path)
    for metin_color in metins_colors:
        mark_objects(test, metin_color)

    viewImage(test)


def main():
    for i in range(1, 12):
        mark_metins(f'images/{i}.png')
    # test = cv.imread('images/8.png')
    # hsv_img = cv.cvtColor(test, cv.COLOR_RGB2HSV)
    # curr_mask = cv.inRange(hsv_img, metin_lv20[0], metin_lv20[1])
    # res = cv.bitwise_and(hsv_img, hsv_img, mask = curr_mask)
    # # viewImage(res)
    # #
    # hsv_img[curr_mask > 0] = (metin_lv20[1])
    # hsv_img[curr_mask == 0] = (0, 0, 0)
    # # viewImage(hsv_img) ## 2
    # # ## converting the HSV image to Gray inorder to be able to apply 
    # # ## contouring
    # RGB_again = cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)
    # gray = cv.cvtColor(RGB_again, cv.COLOR_RGB2GRAY)
    # # viewImage(gray) ## 3
    # # # res[curr_mask > 0] = ([255, 255, 255])
    # _, threshold = cv.threshold(gray, 90, 255, 0)
    
    # # # viewImage(threshold) ## 4
    # contours, _ =  cv.findContours(threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
  
    # c = max(contours, key = cv.contourArea)
    # x,y,w,h = cv.boundingRect(c)

    # cv.rectangle(test,(x,y),(x+w,y+h),(0,255,0),2)
 
    # viewImage(test)

if __name__ == '__main__':
    main()
