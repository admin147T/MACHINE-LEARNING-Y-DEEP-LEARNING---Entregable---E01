import cv2
import os
import numpy as np

dataPath = './data'
listaPersonas = os.listdir(dataPath)
print('Lista de personas', listaPersonas)

labels = []
facesData = []
label = 0

print ('Leyendo las imagenes ...')
for nameDir in listaPersonas:
    # ruta para cada carpeta que esta dentro de data
    personPath = dataPath + '/' + nameDir

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)

        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))

    label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Entrenando para reconocer rostros
print("Entrenando modelo...")
face_recognizer.train(facesData, np.array(labels))

#almacenar  al modelo obtenido
face_recognizer.write('modelo.xml')
print("Modelo Almacenado")