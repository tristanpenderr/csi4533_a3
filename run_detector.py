#----------------------------------------------------------------------------
# Par  : Fahed Hassanat
# Date: 2022-03-01 
# Pour : CSI4533, professeur Laganiere
# Version ='1.0'
# ---------------------------------------------------------------------------

from nn_detector import NN_detector as detector
import cv2
import warnings
warnings.filterwarnings("ignore")

# fixer un seuil pour le score
score_threshold = 0.1

# instancier la classe avec la valeur souhaitée pour "GPU_detect"
det = detector(GPU_detect = False)

# définir le chemin d'une image
path_to_image = "images/000001.jpg"

# exécuter l'inférence sur l'image
detections = det.detect(path_to_image)

# charger l'image dans OpenCV pour la manipuler
img = cv2.imread(path_to_image)

# boucle sur les détections et filtre les scores faibles
for i in range(0, len(detections["boxes"])):
	score = detections["scores"][i]
	id = int(detections["labels"][i])

    # ici, 3 correspond à une voiture, 1 à une personne et 6 à un autobus
	if score > score_threshold and (id == 3 or id == 6 or id == 1):
            box = detections["boxes"][i]       
            (x1, y1, x2, y2) = box.astype("int")

            # dessiner les boîtes englobantes sur l'image
            cv2.rectangle(img, (x1, y1), (x2, y2), [0,0,255], 2)



# afficher l'image
cv2.imshow("Output", img)
cv2.waitKey(0)



