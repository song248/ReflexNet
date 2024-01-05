import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from OTSNet import OTSNet
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
model_path = '/userHome/userhome3/taekyung/OTSNet/saved_model/OTSNet_reflex_mask.pt'
# model = OTSNet()
model = nn.DataParallel(OTSNet())
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model = model.eval()


transform = transforms.Compose([
    # transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_folder  = '/userHome/userhome3/taekyung/OTSNet/Testset/images/'
save_path = '/userHome/userhome3/taekyung/OTSNet/results/'

for image_filename in os.listdir(image_folder):
    print(image_filename)
    if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
        image_path = os.path.join(image_folder , image_filename)
        image = Image.open(image_path)
        image = image.convert('RGB')
        input_image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_image)
        prediction = torch.sigmoid(prediction)
        pred_draw = prediction.clone().detach()   
        img_numpy = pred_draw.cpu().detach().numpy()[0][0]
        
        img_numpy[img_numpy >= 0.5] = 255
        img_numpy[img_numpy < 0.5] = 0
        
        rgb_img = cv2.cvtColor(img_numpy, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite('./results/'+image_filename, rgb_img)
            
        # predicted_mask = prediction.argmax(1).squeeze().cpu().numpy()
        # mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
        # mask_image.save(save_path+image_filename)
