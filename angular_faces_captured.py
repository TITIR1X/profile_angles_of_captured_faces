from cgi import test
from email.mime import image
from PIL import Image
import cv2, os, shutil, webbrowser
from unittest import result
import mediapipe as mp
from collections import Counter
import operator

os.system('cls')
print("""
 ./Angular_rostros.py
  ____              _______ _   ______ _       ___ __    __
 |  _ \            /__   __(_)/__   __(_) _ __|_| |\ \\  / /
 | |_) |_   _         | |   _    | |   _ | '__| | | \ \\/ /
 |  _ <| | | |        | | 0| |   | |  | || |    | |  \  \\
 | |_) | |_| |        | | /| |   | |  | || |    | | / /\ \\
 |____/ \__, |        |_| /|_|   |_|  |_||_|    |_|/_/  \_\\
         __/ |                                               
        |___/                           
""")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

mostrar_differences = input('Add coordinates in the title of each image? It will be useful if you want to calibrate the program: (Y/N) ').upper()

imagesPath = input('Folder with images to angular: ')
imagesPath = imagesPath.replace("\\", '/')

titulo = input(f'Output folder: ./')
output_folder = f'Ordered_Angles_{titulo}'

try:
    os.system(f'md {output_folder}')
    os.makedirs(f'{output_folder}/3')
    os.makedirs(f'{output_folder}/6')
    os.makedirs(f'{output_folder}/9')
    os.makedirs(f'{output_folder}/others/13')
    os.makedirs(f'{output_folder}/others/21')
    os.makedirs(f'{output_folder}/others/34')
    os.makedirs(f'{output_folder}/others/89')
    os.makedirs(f'{output_folder}/others/144')
    os.makedirs(f'{output_folder}/others/others')
except:pass

index_list=[227,116, 345,447, 4,8]

imagesPathList = os.listdir(imagesPath)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

confidence = 1
count = 0
count2 = 0

while count2 < len(imagesPathList):
    for imageName in imagesPathList:
        with mp_face_mesh.FaceMesh(
            static_image_mode = True,
            max_num_faces = 1,
            min_detection_confidence = confidence) as face_mesh:
            try:
                image = cv2.imread(f"{imagesPath}/{imagesPathList[count]}")
                height, width, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(image_rgb)

                if result.multi_face_landmarks is not None:
                    for face_landmarks in result.multi_face_landmarks:
                        for index in index_list:
                            # Obtener coordenadas de los puntos de interés
                            x_227 = int(face_landmarks.landmark[227].x * width)
                            x_116 = int(face_landmarks.landmark[116].x * width)
                            x_345 = int(face_landmarks.landmark[345].x * width)
                            x_447 = int(face_landmarks.landmark[447].x * width)
                            x_4 = int(face_landmarks.landmark[4].x * width)
                            x_8 = int(face_landmarks.landmark[8].x * width)

                            # Calcular ángulos y diferencias
                            angulo_227_116 = x_116 - x_227
                            angulo_345_447 = x_447 - x_345
                            angle_diff_side = angulo_227_116 - angulo_345_447
                            diferencia_angulo = x_4 - x_8

                            if 'S' or 'Y' in mostrar_differences:
                                differences = f'{x_4}-{x_8}={diferencia_angulo}    {angulo_227_116}-{angulo_345_447}={angle_diff_side}    '   
                            if 'N' in mostrar_differences:
                                differences = ''

                        # Calcular ángulo de rotación y aplicar cambio de perspectiva
                        angle = -diferencia_angulo * 0.22223
                        (h, w) = image.shape[:2]
                        center = (w / 2, h / 2)
                        scale = 1
                        M = cv2.getRotationMatrix2D(center, angle, scale)
                        imageOut = cv2.warpAffine(image, M, (w, h))

                        if angle_diff_side <= 3 and angle_diff_side >= -3:
                            cv2.imwrite(f'{output_folder}/3/{differences}{round(confidence,2)}_{imagesPathList[count]}', imageOut)
                        elif angle_diff_side <= 6 and angle_diff_side >= -6:
                            cv2.imwrite(f'{output_folder}/6/{differences}{round(confidence,2)}_{imagesPathList[count]}', imageOut)
                        elif angle_diff_side <= 9 and angle_diff_side >= -9:
                            cv2.imwrite(f'{output_folder}/9/{differences}{round(confidence,2)}_{imagesPathList[count]}', imageOut)
                        elif angle_diff_side <= 13 and angle_diff_side >= -13:
                            cv2.imwrite(f'{output_folder}/others/13/{differences}{round(confidence,2)}_{imagesPathList[count]}', imageOut)
                        elif angle_diff_side <= 21 and angle_diff_side >= -21:
                            cv2.imwrite(f'{output_folder}/others/13/{differences}{round(confidence,2)}_{imagesPathList[count]}', imageOut)
                        elif angle_diff_side <= 34 and angle_diff_side >= -34:
                            cv2.imwrite(f'{output_folder}/others/34/{differences}{round(confidence,2)}_{imagesPathList[count]}', imageOut)
                        elif angle_diff_side <= 89 and angle_diff_side >= -89:
                            cv2.imwrite(f'{output_folder}/others/89/{differences}{round(confidence,2)}_{imagesPathList[count]}', imageOut)
                        elif angle_diff_side <= 144 and angle_diff_side >= -144:
                            cv2.imwrite(f'{output_folder}/others/144/{differences}{round(confidence,2)}_{imagesPathList[count]}', imageOut)
                        else:
                            cv2.imwrite(f'{output_folder}/others/others/{x_4}-{x_8}={diferencia_angulo}    {angulo_227_116}-{angulo_345_447}={angle_diff_side}    {round(confidence,2)}_{imagesPathList[count]}', imageOut)
                    print(f'[{count} de {len(imagesPathList)}] {round(confidence,2)}_{imagesPathList[count]} ..ok')
                    confidence = 1.0
                    count += 1
                    count2 += 1
                    break

                else:
                    confidence -=0.015

                    if confidence <= 0.1:
                        confidence = 1.0
                        count+= 1
            except IndexError:
                print('Trabajo finalizado.')
                exit()

webbrowser.open(os.path.realpath(output_folder))