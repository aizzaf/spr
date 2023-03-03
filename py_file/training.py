import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
import multiprocessing
import functools
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def read_csv(img,data,lbl,rot,id):
    rows = open('dataset4_csv/ID_'+str(id)+'.csv').read().strip().split('\n')
    for row in rows:
        row = row.split(',')
        (x,y,label,rotation,pathimg) = row
        image =  load_img(pathimg, target_size=(256, 256))
        image = img_to_array(image)
        img.append(image)
        data.append([x,y])
        lbl.append(label)
        rot.append(rotation)

if __name__ == '__main__':
    
    manager = multiprocessing.Manager()
    img = manager.list()
    data = manager.list()
    lbl = manager.list()
    rot = manager.list()

    id = list(range(1,9)) #change here

    partial_read_csv = functools.partial(read_csv, img, data, lbl, rot)

    pool = multiprocessing.Pool()
    pool.map(partial_read_csv, id)

pool.close()
pool.terminate()

img = np.array(img,dtype='float32')
data = np.array(data,dtype='float32')
lbl = np.array(lbl)
rot = np.array(rot)
lb = LabelBinarizer()
lbl = lb.fit_transform(lbl)
rot = lb.fit_transform(rot)

split = train_test_split(img, lbl, rot, data, test_size=0.1)
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainRotations, testRotations) = split[4:6]
(trainCoordinates, testCoordinates) = split[6:8]
split = train_test_split(trainImages, trainLabels, trainRotations, trainCoordinates, test_size=0.1)
(trainImages, valImages) = split[:2]
(trainLabels, valLabels) = split[2:4]
(trainRotations, valRotations) = split[4:6]
(trainCoordinates, valCoordinates) = split[6:8]

model = load_model('super3.h5', compile=False)

model.compile(loss={
	'label': 'categorical_crossentropy',
	'rotation': 'categorical_crossentropy',
	'coordinate': 'mse',
}, optimizer='adam', metrics=['accuracy'])

hist = model.fit(
	trainImages, {
	'label': trainLabels,
	'rotation': trainRotations,
	'coordinate': trainCoordinates,
},
	validation_data= (valImages, {
	'label': valLabels,
	'rotation': valRotations,
	'coordinate': valCoordinates,
}),
	batch_size=32,
	epochs=20)

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.savefig('loss3.PNG')

fig = plt.figure()
plt.plot(hist.history['label_accuracy'], color='teal', label='label_accuracy')
plt.plot(hist.history['val_label_accuracy'], color='orange', label='val_label_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.savefig('label3acc.PNG')

fig = plt.figure()
plt.plot(hist.history['rotation_accuracy'], color='teal', label='rotation_accuracy')
plt.plot(hist.history['val_rotation_accuracy'], color='orange', label='val_rotation_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.savefig('rotation3acc.PNG')

fig = plt.figure()
plt.plot(hist.history['coordinate_accuracy'], color='teal', label='coordinate_accuracy')
plt.plot(hist.history['val_coordinate_accuracy'], color='orange', label='val_coorrdinate_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.savefig('coordinate3acc.PNG')

yhat = model.predict(testImages)
y_true = [testLabels,testRotations,testCoordinates]

f = open( 'testing.txt', 'w' )
f.write( 'label = ' + str(accuracy_score(np.amax(y_true[0],axis=1),np.amax(yhat[0].round(),axis=1))) + '\n' )
f.write( 'rotation = ' + str(accuracy_score(np.amax(y_true[1],axis=1),np.amax(yhat[1].round(),axis=1))) + '\n' )
f.write( 'coordinate = ' + str(r2_score(y_true[2],yhat[2])) + '\n' )
f.close()




