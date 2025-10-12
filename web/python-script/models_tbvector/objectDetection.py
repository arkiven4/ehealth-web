import torch
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import os
from ultralytics import YOLO

class ModelObject:
    def __init__(self, model_path="./python-script/models_tbvector/YOLO_Cough.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"error {e}")
            raise
        
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.resize((640, 640), Image.Resampling.LANCZOS)
        
        return image
    
    def detect(self, image_path, output_folder="results"):
        os.makedirs(output_folder, exist_ok=True)
        
        processed_image = self.preprocess_image(image_path)
        
        input_filename = Path(image_path).stem
        
        counter = 1
        unique_folder = "detect"
        full_output_folder = os.path.join(output_folder, unique_folder)
        
        while os.path.exists(full_output_folder):
            unique_folder = f"detect_{counter}"
            full_output_folder = os.path.join(output_folder, unique_folder)
            counter += 1
        
        results = self.model.predict(
            source=processed_image,
            save=True,
            project=output_folder,
            name=unique_folder,
            exist_ok=False,
            conf=0.5,
            iou=0.45
        )
        
        output_file_path = None
        if os.path.exists(full_output_folder):
            original_name = Path(image_path).name
            potential_output = os.path.join(full_output_folder, original_name)
            
            if os.path.exists(potential_output):
                output_file_path = potential_output
            else:
                files = [f for f in os.listdir(full_output_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    output_file_path = os.path.join(full_output_folder, files[0])
        
        return results, full_output_folder, output_file_path
    
    def process_image(self, input_path, output_folder="results"):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file gak ketemu: {input_path}")
            
        results, output_folder_path, output_file_path = self.detect(input_path, output_folder)
        return results, output_folder_path, output_file_path
    
def main():
    if len(sys.argv) != 2:
        print("ERROR: Usage: python object.py <image_filename>", file=sys.stderr)
        sys.exit(1)
    
    image_filename = sys.argv[1]
    
    base_path = "/usr/src/app/public/uploads/foto_indikasi/"  # path default buat gambar
    image_path = os.path.join(base_path, image_filename)
    
    if not os.path.exists(image_path):
        alternative_paths = [
            "images/",
            "test_images/",
            "data/",
            "input/",
            "./"
        ]
        
        for alt_path in alternative_paths:
            test_path = os.path.join(alt_path, image_filename)
            if os.path.exists(test_path):
                image_path = test_path
                break
    
    if not os.path.exists(image_path):
        print(f"error Image file not found: {image_filename}", file=sys.stderr)
        sys.exit(1)
    
    output_folder = os.path.join(base_path, "results")          
    detector = ModelObject()
    
    try:
        results, output_folder_path, output_file_path = detector.process_image(image_path, output_folder)
        print(f"save to folder: {output_folder_path}")
        if output_file_path:
            # print output file path
            print(output_file_path)
        else:
            print("No output file path found.")
            
            
    except Exception as e:
        print(f"error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
