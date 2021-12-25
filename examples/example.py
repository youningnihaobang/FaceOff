from FaceOff.AFR import load_data, Attack
from PIL import Image

# Load the data.  This will detect and resize the faces
print("load input")
inputs = load_data('./faces/input/')
print("load target")
targets = load_data('./faces/target/')
print("load target")
masks = load_data("./faces/mask/")
# Initialize the Attack object with 
adversarial = Attack(inputs[0], targets[0],masks[0] ,optimizer='adam')

# Perform adversarial training
adversarial_tensor, mask_tensor, img = adversarial.train(detect=True, verbose=True)

# Show the image with mask applied
img.show()