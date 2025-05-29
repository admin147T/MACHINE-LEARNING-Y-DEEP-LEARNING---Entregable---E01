#pip install imutils
import cv2
import os
import imutils
from camera import getcamera

print('Escribe tu nombre:')
personName = input()
dataPath = './data'
personPath = dataPath + '/' + personName

if os.path.exists(personPath):
    print('Persona ya registra, sobreescribiendo datos ...')
else:
    os.makedirs(personPath)
    print('Nueva persona, capturando datos...')

#abrir camara
camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

#cargar el detector de rostros
faceClassif = cv2.CascadeClassifier("rostros.xml")

contador = 0

while True:
    # tomar fotografia
    ret, frame = cap.read()

    if not ret:
        break

    # cambiando el tamaño de la foto
    frame = imutils.resize(frame, width=640)

    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(120, 120),
                                         maxSize=(1000, 1000))
    
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #COPIA DE LA IMAGEN
        auxframe = frame.copy()

        # OBTENEMOS EL RECUADRO DEL ROSTTRO
        rostro = auxframe[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150,150),interpolation=cv2.INTER_CUBIC)

        #guardar el rostro como imagen
        cv2.imwrite(personPath + '/rostro_{}.jpg'. format(contador), rostro)
        print('rostro_{}.jpg'.format(contador)+ ' guardado')

        contador = contador + 1

    cv2.imshow('Mi cara', frame)

    if contador >=3 or cv2.waitKey(1) & 0xFF == ord('q'):
        break


#cierre camara y ventanas
cv2.destroyAllWindows()
cap.release()