import torch
from torchvision import models, transforms
from PIL import Image
import requests
import os
from io import BytesIO

# Load ResNet-50 pretrained model
model = models.resnet50(pretrained=True)
model.eval()  # Set to evaluation mode

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]
    )
])



 #while loop for detecting pictures again and again 
while 1:
    # clear the terminal screen
    os.system("clear")
    print(r''' 
    			 _   _       ____ ____ _____    _ 
    			| | | | | | / ___/ ___|__  / \ | |
    			| |_| | | | \___ \___ \ / /|  \| |
    			|  _  | |_| |___) |__) / /_| |\  |
    			|_| |_|\___/|____/____/____|_| \_|

              		    github.com/hussainwaseer

                        IMAGE DETECTOR MODEL                         
    ''')

    inputForSearch = int(input("How do you want to give the picture\n1. for url\n2, for giving picture locally\n3. for exit\nEnter your choice: "))
    getImage=""
    img=""
    if inputForSearch==1:
        getImage=input("Enter the url of the picture: ")
        response=requests.get(getImage)
        img=Image.open(BytesIO(response.content))
    elif inputForSearch==2:
        getImage = input("Enter the picture name with its extension: ")
        img = Image.open(getImage)
    elif inputForSearch==3:
        os.system("clear")
        print("Ending the session Good Bye....")
        input("Press any key to exit")
        break
    else:
        print("Wrong input.....")
    # Preprocess the image
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # create mini-batch of 1 image

    # Load ImageNet labels
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(LABELS_URL).text.strip().split("\n")

    # Move to CPU or GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    # Inference
    with torch.no_grad():
        output = model(input_batch)

   
    # Get predicted label
    _, predicted_idx = torch.max(output, 1)
    print("Predicted label:", labels[predicted_idx.item()])

    input("Enter any key to continue......")
