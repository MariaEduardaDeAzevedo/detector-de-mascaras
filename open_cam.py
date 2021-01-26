import cv2 as cv
import functions

cam = cv.VideoCapture(0)
file_name = "haarcascade_frontalface_alt2.xml"
classifier = cv.CascadeClassifier(f"{cv.haarcascades}/{file_name}")

dataframe = functions.load_dataframe()

X_train, X_test, y_train, y_test = functions.train_test(dataframe)
pca = functions.pca_model(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

knn = functions.knn(X_train, y_train)

label = {
    0: "Sem mascara",
    1: "Com mascara"
}

while True:
    status, frame = cam.read()

    if not status:
        break

    if cv.waitKey(1) & 0xff == ord('q'):
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray)

    for x,y,w,h in faces:
        gray_face = gray[y:y+h, x:x+w]

        if gray_face.shape[0] >= 200 and gray_face.shape[1] >= 200:
            gray_face = cv.resize(gray_face, (160,160))
            vector = pca.transform([gray_face.flatten()])

            pred = knn.predict(vector)[0]
            classification = label[pred]

            if pred == 0:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
            elif pred == 1:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

            cv.putText(frame, classification, (x - 20,y + h + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)
            cv.putText(frame, f"{len(faces)} rostos identificados",(20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)

    cv.imshow("Cam", frame)


    
