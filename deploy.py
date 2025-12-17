#General
import os
import numpy as np
import pandas as pd

#Feature extraction and Model
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from xgboost import XGBRegressor

#Image Processing and Display
from tqdm import tqdm
import warnings
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io 
warnings.filterwarnings('ignore')

def load_and_process_image(image_path, device, mtcnn, resnet):
    try:
        img = Image.open(image_path)
        img_cropped = mtcnn(img)
        
        if img_cropped is None:
            print(f"No face detected in {image_path}")
            return None
            
        img_cropped = torch.unsqueeze(img_cropped, 0).to(device)
        with torch.no_grad():
            features = resnet(img_cropped)
        return features.cpu().numpy().flatten()
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None
        
def extract_features(base_path, max_persons=1000):
    print("here")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    mtcnn = MTCNN(device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    front_path = os.path.join(base_path, 'front')
    front_files = sorted(os.listdir(front_path))

    
    if max_persons: front_files = front_files[:max_persons]
    
    all_features = []
    processed_files = []
    
    for front_file in tqdm(front_files, desc="Processing images"):
        side_file = front_file  # Same filename in side folder
        
        front_features = load_and_process_image(
            os.path.join(front_path, front_file),
            device, mtcnn, resnet
        )
        #print("front-features")
        print(front_features)
        # print("Hi")
        
        side_features = load_and_process_image(
            os.path.join(base_path, 'side', side_file),
            device, mtcnn, resnet
        )


        print(side_features)
        if front_features is not None and side_features is not None:
            combined_features = np.concatenate([front_features, side_features])
            all_features.append(combined_features)
            processed_files.append(front_file)


    
    # Create feature column names
    front_cols = [f'front_feature_{i}' for i in range(512)]  # FaceNet outputs 512-D vectors
    side_cols = [f'side_feature_{i}' for i in range(512)]
    all_cols = front_cols + side_cols
    
    df = pd.DataFrame(all_features, columns=all_cols)
    df.insert(0, 'id', processed_files)
    
    return df
    
    

