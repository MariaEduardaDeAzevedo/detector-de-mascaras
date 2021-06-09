import cv2 as cv
import functions
import os

cam = cv.VideoCapture(0) #Iniciando a WebCam

dataframe = functions.load_dataframe() #Carregando dataframe com as imagens para treinamento

X_train, y_train = functions.train_test(dataframe) #Dividindo conjuntos de treino e teste
pca = functions.pca_model(X_train) #Modelo PCA para extração de features da imagem

X_train = pca.transform(X_train) #Conjunto de treino com features extraídas
#X_test = pca.transform(X_test) #Conjunto de teste com features extraídas

knn = functions.knn(X_train, y_train) #Treinando modelo classificatório KNN

#Rótulo das classificações
label = {
    0: "Sem mascara",
    1: "Com mascara"
}

#Abrindo a webcam...
while True:
    status, frame = cam.read() #Lendo a imagem e extraindo frame

    if not status:
        break

    if cv.waitKey(1) & 0xff == ord('q'):
        break
    
    #Transformando a imagem em escala de cinza
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Detectando faces na imagem
    height, width, _ = frame.shape
    pt1, pt2 = ((width//2) - 100, (height//2) - 100), ((width//2) + 100, (height//2) + 100)
    region = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    gray_face = cv.cvtColor(region, cv.COLOR_BGR2GRAY)

    gray_face = cv.resize(gray_face, (160,160)) #Redimensionando
    vector = pca.transform([gray_face.flatten()]) #Extraindo features da imagem
    
    pred = knn.predict(vector)[0]
    classification = label[pred]

    color = (0,0,255)

    if pred == 1:
        color = (0,255,0)

    #Escrevendo classificação e quantidade de faces vistas
    cv.putText(frame, classification, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)

    cv.rectangle(frame, pt1, pt2, color,thickness=3)
    #Mostrando o frame
    cv.imshow("Cam", frame)
