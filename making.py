import cv2
import numpy as np
import pandas as pd
import multiprocessing

def translate(path,x,y):
  img = cv2.imread(path)
  transMat = np.float32([[1,0,x],[0,1,y]])
  dimension = (img.shape[1], img.shape[0])
  hasil = cv2.warpAffine(img,transMat,dimension)
  cv2.imwrite(path.split('.')[0]+'trans'+str(x)+'_'+str(y)+'.jpg',hasil)

  df = pd.DataFrame({
        'x': x/img.shape[1],
        'y': y/img.shape[0],
        'label': path.split('.')[0].split('/')[1],
        'rot': path.split('.')[0].split('/')[2],
        'path': path.split('.')[0]+'trans'+str(x)+'_'+str(y)+'.jpg',   
      },index=[0])
  df.to_csv('dataset4_csv/'+path.split('.')[0].split('/')[1]+'.csv', mode='a', index=False, header=False)

def crop(path,x1,x2,y1,y2):
  img = cv2.imread(path)
  (heightm,widthm) = img.shape[:2]
  hasil = img[y1:y2,x1:x2]
  (heightb,widthb) = hasil.shape[:2]
  cv2.imwrite(path.split('.')[0]+'crop'+str(x1)+'-'+str(x2)+'_'+str(y1)+'-'+str(y2)+'.jpg',hasil)

  df = pd.DataFrame({
        'x': (widthm/2-(x2+x1)/2)/widthb,
        'y': (heightm/2-(y2+y1)/2)/heightb,
        'label': path.split('.')[0].split('/')[1],
        'rot': path.split('.')[0].split('/')[2],
        'path': path.split('.')[0]+'crop'+str(x1)+'-'+str(x2)+'_'+str(y1)+'-'+str(y2)+'.jpg',   
      },index=[0])
  df.to_csv('dataset4_csv/'+path.split('.')[0].split('/')[1]+'.csv', mode='a', index=False, header=False)

def id_translate(id):
  for a in range(0,337,24):
    for x in range(-200,201,50):
      for y in range(-200,201,50):
        translate('dataset4_image/ID_'+str(id)+'/'+str(a)+'.jpg',x,y)

def id_crop(id):
  for a in range(0,337,24):
    for x1 in range(100,201,50):
      for x2 in range(2900,3001,50):
        for y1 in range(100,201,50):
          for y2 in range(2300,2401,50):
            crop('dataset4_image/ID_'+str(id)+'/'+str(a)+'.jpg',x1,x2,y1,y2)

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    pool.map(id_translate, range(1,9))
    pool.map(id_crop, range(1,9))

pool.close()
pool.terminate()