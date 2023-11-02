import torch
from torchvision import models
import numpy as np
import cv2

# specify image dimension
IMAGE_SIZE = 224
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# determine the device we will be using for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# specify path to the ImageNet labels
IN_LABELS = "ilsvrc2012_wordnet_lemmas.txt"

imagePath = "1.jpg"

#preTrainedModel = "vgg16"
#preTrainedModel = "vgg19"
#preTrainedModel = "inception"
preTrainedModel = "resnet"
#preTrainedModel = "densenet"


def preprocess_image(image):
	# swap the color channels from BGR to RGB, resize it, and scale
	# the pixel values to [0, 1] range
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
	image = image.astype("float32") / 255.0
	# subtract ImageNet mean, divide by ImageNet standard deviation,
	# set "channels first" ordering, and add a batch dimension
	image -= MEAN
	image /= STD
	image = np.transpose(image, (2, 0, 1))
	image = np.expand_dims(image, 0)
	# return the preprocessed image
	return image

MODELS = {
	"vgg16": models.vgg16(pretrained=True),
	"vgg19": models.vgg19(pretrained=True),
	"inception": models.inception_v3(pretrained=True),
	"densenet": models.densenet121(pretrained=True),
	"resnet": models.resnet50(pretrained=True)
}
# load our the network weights from disk, flash it to the current
# device, and set it to evaluation mode
print("[INFO] loading {}...".format(preTrainedModel))
model = MODELS[preTrainedModel].to(DEVICE)
model.eval()

print("[INFO] loading image...")
image = cv2.imread(imagePath)
orig = image.copy()
image = preprocess_image(image)
# convert the preprocessed image to a torch tensor and flash it to
# the current device
image = torch.from_numpy(image)
image = image.to(DEVICE)
# load the preprocessed the ImageNet labels
print("[INFO] loading ImageNet labels...")
imagenetLabels = dict(enumerate(open(IN_LABELS)))

# classify the image and extract the predictions
print("[INFO] classifying image with '{}'...".format(preTrainedModel))
logits = model(image)
probabilities = torch.nn.Softmax(dim=-1)(logits)
sortedProba = torch.argsort(probabilities, dim=-1, descending=True)
# loop over the predictions and display the rank-5 predictions and
# corresponding probabilities to our terminal
for (i, idx) in enumerate(sortedProba[0, :5]):
	print("{}. {}: {:.2f}%".format
		(i, imagenetLabels[idx.item()].strip(),
		probabilities[0, idx.item()] * 100))

# draw the top prediction on the image and display the image to
# our screen
(label, prob) = (imagenetLabels[probabilities.argmax().item()],
	probabilities.max().item())
cv2.putText(orig, "Label: {}, {:.2f}%".format(label.strip(), prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)