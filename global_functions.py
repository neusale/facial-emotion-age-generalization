import gdown
import glob
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from PIL import Image
import tensorflow as tf
import tensorflow.keras.utils as Utils
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import seaborn as sns
from matplotlib import pyplot
import scikitplot
from sklearn.metrics import classification_report
import collections
import altair as alt


class filesStru:
    def __init__(self, filesName, filesClass):
        self.files = filesName
        self.files_class = filesClass

def instance_cascade_classifier():
    return cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

def instance_detector_mtcnn():
    return MTCNN()

def download_extract(url, output, to):
    gdown.download(url, output, quiet=False)
    gdown.extractall(output, to)
    
def obter_imagens(diretorio, extensao='jpg'):
   array_arquivos = glob.glob(diretorio+'/*.'+extensao)
   return array_arquivos

def obter_modelos_salvos(dataSetTreino, extensao='h5'):
   array_arquivos = glob.glob('*_' + dataSetTreino + '.' + extensao)
   return array_arquivos
   
def obter_classes(quantidade, indice_emocao):
   array_classes = np.full(quantidade,indice_emocao).astype('int')
   return array_classes    
   
def getFiles(dataset, crop=False, extensao='jpg'):
    #emotions_labels = {0:'raiva', 1:'aversão', 2:'medo', 3:'alegria', 4: 'tristeza', 5: 'surpresa', 6: 'desprezo'}
    drive_name = dataset + '_Angry'
    if crop==True:
        drive_name += '_cropped'
    files_raiva = obter_imagens(drive_name, extensao)
    filesclass_raiva = obter_classes(len(files_raiva), 0)
    print('Raiva: '+ str(len(files_raiva)))

    drive_name = dataset + '_Fear'
    if crop==True:
        drive_name += '_cropped'
    files_medo = obter_imagens(drive_name, extensao)
    filesclass_medo = obter_classes(len(files_medo), 1)
    print('Medo: ' + str(len(files_medo)))

    drive_name = dataset + '_Happy'
    if crop==True:
        drive_name += '_cropped'
    files_alegria = obter_imagens(drive_name, extensao)
    filesclass_alegria = obter_classes(len(files_alegria), 2)
    print('Alegria: ' + str(len(files_alegria)))

    drive_name = dataset + '_Sad'
    if crop==True:
        drive_name += '_cropped'
    files_tristeza = obter_imagens(drive_name, extensao)
    filesclass_tristeza = obter_classes(len(files_tristeza), 3)
    print('Tristeza: ' + str(len(files_tristeza)))   
    
    files = np.concatenate((files_raiva, files_medo, files_alegria, files_tristeza), axis=0)
    files_class = np.concatenate((filesclass_raiva, filesclass_medo, filesclass_alegria, filesclass_tristeza), axis=0)
    print('Total imagens: ' + str(len(files)))
    print('Total classes: ' + str(len(files_class)))
    return filesStru(files, files_class) 
    
    
def createModel(modelName, img_height, img_width, modelType='m1', num_classes=4):
    indTrainable = -16
    layerInd = -3
    if modelName == 'MobileNetV2':
        model = tf.keras.applications.MobileNetV2(weights = 'imagenet', include_top = False, input_shape = (img_height, img_width, 3))
    if modelName == 'DenseNet121':
        model = tf.keras.applications.DenseNet121(weights = 'imagenet', include_top = False, input_shape = (img_height, img_width, 3))    
    if modelName == 'DenseNet201':
        model = tf.keras.applications.DenseNet201(weights = 'imagenet', include_top = False, input_shape = (img_height, img_width, 3))    
    if modelName == 'ResNet50V2':
        model = tf.keras.applications.ResNet50V2(weights = 'imagenet', include_top = False, input_shape = (img_height, img_width, 3))    
    if modelName == 'ResNet101V2':
        model = tf.keras.applications.ResNet101V2(weights = 'imagenet', include_top = False, input_shape = (img_height, img_width, 3))    
    if modelName == 'ResNet152V2':
        model = tf.keras.applications.ResNet152V2(weights = 'imagenet', include_top = False, input_shape = (img_height, img_width, 3))    
    if modelName == 'VGG16':
        model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (img_height, img_width, 3))    
        indTrainable = -3
        layerInd = -2

    model.trainable = False
    for layer in model.layers[indTrainable:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    if modelType == 'm1':
        head = build_model_m1(model, num_classes, layerInd)
        newModel = Model(inputs = model.input, outputs = head)
        newModel.summary(show_trainable=True)        

    if modelType == 'm2':
        head = build_model_m2(model, num_classes, layerInd)
        newModel = Model(inputs = model.input, outputs = head)
        newModel.summary(show_trainable=True)        
        
    return newModel    

def compileAndTrain(modelName, modelType, dataset, modelObject, trainGenerator, validGenerator, XTrain, classWeights):
    batchSize = 20
    epochs = 30
    optims = [optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999),]

    early_stopping = EarlyStopping(monitor = 'val_accuracy',
                               min_delta = 0.00005,
                               patience = 11,
                               verbose = 1,
                               restore_best_weights = True,)

    lr_scheduler = ReduceLROnPlateau(monitor = 'val_accuracy',
                                 factor = 0.5,
                                 patience = 7,
                                 min_lr = 1e-7,
                                 verbose = 1,)

    callbacks = [early_stopping,lr_scheduler,]            
    
    modelObject.compile(loss = 'categorical_crossentropy',
              optimizer = optims[0],
              metrics = ['accuracy'])
              
    history = modelObject.fit(trainGenerator,
                       validation_data = validGenerator,
                       batch_size = batchSize,
                       steps_per_epoch = len(XTrain) / batchSize,
                       epochs = epochs,
                       callbacks = callbacks,
                       class_weight=classWeights
                   )   
    model_yaml = modelObject.to_json()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    modelObject.save(modelName + "_" + modelType + "_" + dataset + ".h5")
    modelObject.save_weights(modelName + "_" + modelType + "_" + dataset + "_weights.h5")                              
    
    #show graph
    sns.set()
    fig = pyplot.figure(0, (12, 4))

    ax = pyplot.subplot(1, 2, 1)
    sns.lineplot(x=history.epoch, y=history.history['accuracy'], label='train')
    sns.lineplot(x=history.epoch, y=history.history['val_accuracy'], label='valid')
    pyplot.title('Accuracy')
    pyplot.tight_layout()

    ax = pyplot.subplot(1, 2, 2)
    sns.lineplot(x=history.epoch, y=history.history['loss'], label='train')
    sns.lineplot(x=history.epoch, y=history.history['val_loss'], label='valid')
    pyplot.title('Loss')
    pyplot.tight_layout()

    pyplot.savefig("resultados/epoch_history_" + modelName + "_" + modelType + "_" + dataset + ".png")
    pyplot.show()
    
    return modelObject
    
def test_model(modelName, modelType, dataset, model, XTest, yTest, emotionsLabels):
    np.set_printoptions(precision=6)	
    print(XTest.shape, yTest.shape)
    XTest = XTest / 255.
    np.set_printoptions(precision=4)

    yhat = np.argmax(model.predict(XTest), axis=1)
    scikitplot.metrics.plot_confusion_matrix(np.argmax(yTest, axis=1), yhat, figsize=(7,7))
    pyplot.savefig("resultados/confusion_matrix_" + modelName + "_" + modelType + "_" + dataset + ".png")

    print(f'total wrong validation predictions: {np.sum(np.argmax(yTest, axis=1) != yhat)}\n\n')
    print(classification_report(np.argmax(yTest, axis=1), yhat, digits=6))

    wrong_predictions(XTest, yTest, yhat, emotionsLabels)
    return yhat    


def classification_report_data(chave, report, reportData):
    lines = report.split('\n')
    del lines[1]
    del lines[5]
    del lines[8]
    for line in lines[1:]:
        row = collections.OrderedDict()
        row_data = line.split()
        row_data = list(filter(None, row_data))
        ind = 0
        row['chave'] = chave
        if (row_data[0] == 'macro' or row_data[0]=='weighted'):
            row['class'] = row_data[0] + ' ' + row_data[1] 
            ind = 2
        else:    
            row['class'] = row_data[0] 
            ind = 1
        if (row_data[0]!='accuracy'):
            row['precision'] = float(row_data[ind])
            row['recall'] = float(row_data[ind+1])
            row['f1_score'] = float(row_data[ind+2])
            row['support'] = int(row_data[ind+3])
        else:
            row['f1_score'] = float(row_data[ind])
            row['support'] = int(row_data[ind+1])
            
        reportData.append(row)
    return reportData 

def apply_prediction(savedModel, testDataset, XTest, yTest, emotionsLabels, reportData):
    np.set_printoptions(precision=6)
    
    print(XTest.shape, yTest.shape)
    XTest = XTest / 255.
    
    nameSavedModel = savedModel
    nameSavedWeights = savedModel.replace(".h5", "_weights.h5")
    name = savedModel.replace(".h5", "_TestOn" + testDataset)
    trainedModel = load_model(nameSavedModel)
    trainedModel.load_weights(nameSavedWeights)
    
    yhat = np.argmax(trainedModel.predict(XTest), axis=1)
    scikitplot.metrics.plot_confusion_matrix(np.argmax(yTest, axis=1), yhat, figsize=(7,7))
    pyplot.savefig("resultados/predicoes/confusion_matrix_" + name + ".png")

    print(f'total wrong validation predictions: {np.sum(np.argmax(yTest, axis=1) != yhat)}\n\n')
    report = classification_report(np.argmax(yTest, axis=1), yhat, digits=6)
    classificationReportData = classification_report_data(name, report, reportData)
    print(report)

    wrong_predictions(XTest, yTest, yhat, emotionsLabels)
    return classificationReportData        

def reportDataToDataFrame(reportData):    
    df = pd.DataFrame.from_dict(reportData)
    df.set_index('class', inplace=True)
    return df
    
            
def build_model_m1(bottom_model, classes, layerInd=-3):
    model = bottom_model.layers[layerInd].output
    model = Flatten()(model)
    model = Dense(256, activation = 'relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(classes, activation = 'softmax', name = 'out_layer')(model)
    return model
    
def build_model_m2(bottom_model, classes, layerInd=-3):
    model = bottom_model.layers[layerInd].output
    model = GlobalAveragePooling2D()(model)
    model = BatchNormalization()(model)
    model = Dense(classes, activation = 'softmax', name = 'out_layer')(model)
    return model        
    
   
def color_bgr_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    return img
    
def color_gray_to_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)    
    return img
   
def color_rgb_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def create_features(files, img_width, img_height):
    img_features = []
    for i, filename in enumerate(files):
        img = cv2.imread(filename)
        img = color_rgb_to_gray(np.array(img))
        img = color_gray_to_rgb(img)
        img = cv2.resize(img, (img_width, img_height))
        img_features.append(img)
    img_features = np.array(img_features)
    return img_features

def some_images(files, fclasses, emotions):
    fig = pyplot.figure(1, (14, 14))
    pos = 0
    r = len(emotions) + 1
    for label in range(r):
        k_img = 0
        j = 0
        while j <= 6 and k_img < len(files):
          if (fclasses[k_img]==label):
              pos += 1
              img = cv2.imread(files[k_img])
              img = color_bgr_to_gray(img)
              img = color_gray_to_rgb(img)
              ax = pyplot.subplot(r, 7, pos)
              ax.imshow(img)
              ax.set_xticks([])
              ax.set_yticks([])
              ax.set_title(emotions[label])
              pyplot.tight_layout()
              j += 1
          k_img += 1

def all_images(files, fclasses, emotions, inicio=0, fim=0):
    if fim==0:
       fim = len(files)
    
    qtd = fim - inicio + 1
    n_cols = 10
    n_rows = int(qtd/n_cols)+1
    fig_width = int(300/n_cols)
    fig_heigth = int(600/n_rows)
    
    fig = pyplot.figure(1, (fig_width, fig_heigth))
    pos = 0
    k_img = inicio
    while k_img < fim:    
       pos += 1
       img = cv2.imread(files[k_img])
       img = color_bgr_to_gray(img)
       img = color_gray_to_rgb(img)
       ax = pyplot.subplot(n_rows, n_cols, pos)
       ax.imshow(img)
       ax.set_xticks([])
       ax.set_yticks([])
       ax.set_title(emotions[fclasses[k_img]])
       pyplot.tight_layout()
       k_img += 1                                           

def wrong_predictions(X, y, yhat, emotions):
    fim = len(y)    
    images = []
    tit = []
    qtd_wrong = 0
    k_img = 0
    while k_img < fim:    
        y_true = np.argmax(y, axis=1)[k_img]
        y_pred = yhat[k_img]
        if (y_pred != y_true):
            images.append(X[k_img])
            tit.append("t:"+ emotions[y_true] + " p:" + emotions[y_pred])
            qtd_wrong += 1
        k_img += 1                                           

    qtd = qtd_wrong
    n_cols = 5
    if qtd <= 6:
        n_cols = 3
        
    #print(qtd, n_cols)    
    n_rows = (qtd + n_cols - 1) // n_cols
    if (n_rows == 1):
        n_rows = 2
    #print(n_rows, n_cols)
    fig, axs = pyplot.subplots(n_rows, n_cols, figsize=(12, 8))
    pos = 0
    k_img = 0
    # Percorre as imagens e as exibe nos subplots
    for i, image in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        #print(i, row, col)
        axs[row, col].imshow(image)
        axs[row, col].axis('off')  # Remove os eixos
        axs[row, col].set_title(tit[i])

    # Garante que não haja subplots vazios
    for i in range(qtd, n_rows * n_cols):
        fig.delaxes(axs.flatten()[i])

    pyplot.tight_layout()
    pyplot.show() 

    
def extract_face(faces, face_image, size=224):
    #face_image = Image.fromarray(face_boundary)
    #face_image = face_image.resize((size, size))
    for i in range(len(faces)):    
        x1, y1, width, height = faces[i]['box']
        x2, y2 = x1 + width, y1 + height
        img_cropped = face_image[y1:y2, x1:x2]
        return img_cropped  
    return None
    
def crop_image_mtcnn(img, detector, size=224):    
    faces = detector.detect_faces(img)
    img_cropped = extract_face(faces, img, size)
    return img_cropped
    
def find_face(img, faces, face_cascade, k, size):
    for i in range(len(faces)):    
        x = faces[i][0]
        y = faces[i][1]
        w = faces[i][2]
        h = faces[i][3]
        img_cropped = img[y:y+h, x:x+w]
        if (verify_cropped_image(img_cropped, face_cascade, k, size)):
           return img_cropped
    return None       
    
def crop_image(img, detector, face_cascade, k=10, size=224):
    minsize = int(round(size/3))
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=k, minSize=(minsize, minsize))    
    img_cropped = find_face(img, faces, face_cascade, k, size)
    
    if (img_cropped is None):
       img_cropped = crop_image_mtcnn(img, detector, size)
       if (img_cropped is None):
           return img
       else:
           return img_cropped    
    else:
       return img_cropped   

def verify_cropped_image(img_crop, face_cascade, k=10, size=224):
    minsize = int(round(size/3))
    faces = face_cascade.detectMultiScale(img_crop, scaleFactor=1.1, minNeighbors=k, minSize=(minsize, minsize))    
    if (len(faces)!=0):
       return True
    return False 

    
def criar_image_features(files, img_width, img_height, crop=False, k=10, size=224):
    img_features = []
    face_cascade = instance_cascade_classifier()
    detector = instance_detector_mtcnn()

    for i, filename in enumerate(files):
        img = cv2.imread(filename)
        if crop==True:
           img_cropped = crop_image(img, detector, face_cascade, k, size)
           img = color_rgb_to_gray(np.array(img_cropped))
        else:
           img = color_rgb_to_gray(np.array(img))
        img = color_gray_to_rgb(img)
        img = cv2.resize(img, (img_width, img_height))
        img_features.append(img)
    img_features = np.array(img_features)
    print(img_features.shape)
    return img_features
    
def amostra_imagens(files, fclasses, emotions, crop=False, k=10, size=224):
    face_cascade = instance_cascade_classifier()
    detector = instance_detector_mtcnn()

    fig = pyplot.figure(1, (14, 14))
    pos = 0
    r = len(emotions) + 1
    for label in range(r):
        k_img = 0
        j = 0
        while j <= 6 and k_img < len(files):
          if (fclasses[k_img]==label):
              pos += 1
              img = cv2.imread(files[k_img])
              img = color_bgr_to_gray(img)
              img = color_gray_to_rgb(img)
              if crop==True:
                 img = crop_image(img, detector, face_cascade, k, size)
              ax = pyplot.subplot(r, 7, pos)
              ax.imshow(img)
              ax.set_xticks([])
              ax.set_yticks([])
              ax.set_title(emotions[label])
              pyplot.tight_layout()
              j += 1
          k_img += 1
          
def amostra_predicoes(X, y, yhat, emotions):
    fig = pyplot.figure(1, (14, 14))
    pos = 0
    r = len(emotions) + 1
    for label in range(r):
        k_img = 0
        j = 0
        while j <= 6 and k_img < len(y):
          y_true = np.argmax(y, axis=1)[k_img]
          y_pred = yhat[k_img]
          if (y_true==label and y_pred != y_true):
              pos += 1
              image = X[k_img]              
              ax = pyplot.subplot(r, 7, pos)
              ax.imshow(image)
              ax.set_xticks([])
              ax.set_yticks([])
              tit = "t:"+ emotions[y_true] + " p:" + emotions[y_pred]
              ax.set_title(tit)
              pyplot.tight_layout()
              j += 1
          k_img += 1
          
def crop_and_see(file, k=10, size=224):
    face_cascade = instance_cascade_classifier()   
    detector = instance_detector_mtcnn()

    img = cv2.imread(file)
    img = color_bgr_to_gray(img)
    img = color_gray_to_rgb(img)
    img_cropped = crop_image(img, detector, face_cascade, k, size)
    ax = pyplot.subplot(2, 2, 1)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("original")
    pyplot.tight_layout()
    ax = pyplot.subplot(2, 2, 2)
    ax.imshow(img_cropped)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("crop")
    pyplot.tight_layout()
                                           
def todas_as_imagens(files, fclasses, emotions, crop=False, k=10, size=224, inicio=0, fim=0):
    face_cascade = instance_cascade_classifier()
    detector = instance_detector_mtcnn()

    if fim==0:
       fim = len(files)
    
    qtd = fim - inicio + 1
    n_cols = 10
    n_rows = int(qtd/n_cols)+1
    fig_width = int(300/n_cols)
    fig_heigth = int(600/n_rows)
    
    fig = pyplot.figure(1, (fig_width, fig_heigth))
    pos = 0
    k_img = inicio
    while k_img < fim:    
       pos += 1
       img = cv2.imread(files[k_img])
       img = color_bgr_to_gray(img)
       img = color_gray_to_rgb(img)
       if crop==True:
          img = crop_image(img, detector, face_cascade, k, size)
       ax = pyplot.subplot(n_rows, n_cols, pos)
       ax.imshow(img)
       ax.set_xticks([])
       ax.set_yticks([])
       ax.set_title(emotions[fclasses[k_img]])
       pyplot.tight_layout()
       k_img += 1                                           
       
       
def save_cropped_image(files, k=10, size=224, inicio=0, fim=0):
    face_cascade = instance_cascade_classifier()
    detector = instance_detector_mtcnn()

    if fim==0:
       fim = len(files)
    
    pos = 0
    k_img = inicio
    while k_img < fim:    
       pos += 1
       filename = files[k_img]
       pos = filename.index('/')
       directory = filename[0:pos]
       filename_crop = filename.replace(directory, directory+"_cropped")
       img = cv2.imread(files[k_img])
       img = color_bgr_to_gray(img)
       img = color_gray_to_rgb(img)
       img = crop_image(img, detector, face_cascade, k, size)
       cv2.imwrite(filename_crop, img)
       k_img += 1                                                  
       
def hist_dataset(name, ext, tit):       
    emotions_labels = {0:'raiva',  1:'medo', 2:'alegria', 3: 'tristeza'}
    filesStru = getFiles(dataset=name, crop=True, extensao=ext)
    files = filesStru.files
    files_class = filesStru.files_class
    data = {'imagem': files, 'emocao': files_class}
    df = pd.DataFrame(data)
    value_counts = df['emocao'].value_counts(normalize=True, ascending=False)
    novo_df = pd.DataFrame({'Cod': value_counts.index, 'Freq': value_counts.values})
    freq_df = pd.DataFrame()
    freq_df['Emoção'] = novo_df['Cod'].map(emotions_labels)
    freq_df['Frequência'] = novo_df['Freq']
    maxF = novo_df['Freq'].max()+0.05
    chart = alt.Chart(freq_df).mark_bar().encode(
       x = alt.X("Frequência", title = "Frequência Relativa", scale=alt.Scale(domain=(0, maxF))),
       y=alt.Y("Emoção"),
       color=alt.Color('Emoção:N', scale=alt.Scale(range=[' #CF8C80', '#8FAD80', '#E6C451', '#647C90']))
    ).properties(height=alt.Step(25))
    text = chart.mark_text(
        align='left',
        baseline='middle',
        fontStyle='bold',
        dx=3
    ).encode(
        text=alt.Text('Frequência:Q', format='.2f')
    )

    return (chart + text).properties(width=300, title=tit)
    
