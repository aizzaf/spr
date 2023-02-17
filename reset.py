import os

for id in os.listdir('dataset4_image'):
    a = os.path.join('dataset4_image',id)
    for filename in os.listdir(a):
        if filename.split('.')[0] != str(0) and filename.split('.')[0] != str(120) and filename.split('.')[0] != str(144) and filename.split('.')[0] != str(168) and filename.split('.')[0] != str(192) and filename.split('.')[0] != str(216) and filename.split('.')[0] != str(24) and filename.split('.')[0] != str(240) and filename.split('.')[0] != str(264) and filename.split('.')[0] != str(288) and filename.split('.')[0] != str(312) and filename.split('.')[0] != str(336) and filename.split('.')[0] != str(48) and filename.split('.')[0] != str(72) and filename.split('.')[0] != str(96):
            os.remove(os.path.join(a,filename))

for id in os.listdir('dataset4_csv'):
    a = os.path.join('dataset4_csv',id)
    os.remove(a)