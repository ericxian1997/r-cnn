# Python version: 2.7.x
import skimage.data
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import selectivesearch
from skimage import transform,data
from overlap import overlap

def main():

    # loading image
    from skimage import io
    img = io.imread('example/dog (1).JPEG')
    # img = skimage.data.astronaut()
    io.imshow(img)
    endpoint = np.arange(4)
    cut_photo = np.arange(154587)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        repeted = False
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        a1 = x
        a2 = y
        a3 = w
        a4 = h
        if w / h > 1.2 or h / w > 1.2:
            continue
        # remove the repeated area
        for x, y, w, h in candidates:
            if overlap(a1,a2,a3,a4,x,y,w,h) > 0.9:
                repeted = True
                break
        if repeted == False:
            candidates.add(r['rect'])
        
    # save the new photo
    i = 1
    for x, y, w, h in candidates:
        print x, y, w, h
        cut_area = img[y:y+h,x:x+w,:]
        io.imsave('C:\Users\eric\selectivesearch\segements\\' + str(i) +'.jpg',cut_area)
        i = i+1
        out = transform.resize(cut_area, (227, 227))
        temp1 = np.array([x,y,w,h])
        temp2 = out
        temp2 = np.array(temp2,dtype=np.float32)
        temp2 = temp2.reshape(1,154587)
        endpoint = np.vstack((endpoint,temp1))
        cut_photo = np.vstack((cut_photo,temp2))

    # save the np.array
    np.save("cut_photo.npy", cut_photo)
    np.save("endpoint_4.npy", endpoint)
    

if __name__ == "__main__":
    main()
