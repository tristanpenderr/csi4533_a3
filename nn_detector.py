#----------------------------------------------------------------------------
# Par  : Fahed Hassanat
# Date: 2022-03-01 
# Pour : CSI4533, professeur Laganiere
# Version ='1.0'
# ---------------------------------------------------------------------------

from torchvision.models import detection
import numpy as np
import torch
import cv2
from timeit import default_timer as timer

class NN_detector:

	def __init__(self, GPU_detect:bool = False):

		# sélectionner le modèle
		self.raw_model = detection.retinanet_resnet50_fpn


		self.gpu_detect = GPU_detect

		# définir les classes, pour référence
		self.classes = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
				'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
				'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
				'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
				'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
				'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
				'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
				'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
				'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
				'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
				'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
				'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
				]

		# initialiser le périphérique de calcul par défaut
		self.device = torch.device("cpu")

		self.set_device()
		self.model = self.set_model()
		
	def set_device(self):
		# définissez le périphérique de calcul en fonction de la valeur transmise de GPU_detect si possible		
		if self.gpu_detect:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			print(self.device)
		device_label = 'GPU' if self.device == torch.device("cuda") else 'CPU'
		print("Using the %s to run the inference" %(device_label))

		
	def set_model(self):
		# définir les paramètres du modèle
		model = self.raw_model(pretrained=True, progress=True, num_classes=len(self.classes), pretrained_backbone=True).to(self.device)

		# définir le modèle comme gelé
		model.eval()

		return model

	def preprocess_image(self, img):
		# prétraiter l'image à transmettre au modèle
		image = cv2.imread(img)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.transpose((2, 0, 1))
		image = np.expand_dims(image, axis=0)
		image = image / 255.0
		image = torch.FloatTensor(image)
		image = image.to(self.device)
		return image

	def detect(self, image):
		image = self.preprocess_image(image)
		start = timer()

		# exécuter le modèle avec l'image sélectionnée
		inference = self.model(image)[0]
		
		
		end = timer()

		print('Inference complete in %.4f milisec' %((end - start)*1000))
	
		inference['boxes'] = inference["boxes"].detach().cpu().numpy()

		
		return inference