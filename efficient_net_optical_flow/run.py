import torch
from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np
import cv2
from IPython.display import display
from model import EfficientNetDetectorOpticalFlow

def load_ir_image(path, target_size=(256, 256)):
    img = Image.open(path).convert("L")  
    img = img.resize(target_size)
    return np.array(img, dtype=np.uint8)

def compute_flow(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )  
    flow = (flow - flow.min()) / (flow.max() - flow.min() + 1e-6)
    flow = flow.transpose(2, 0, 1)  # [2, H, W]
    return flow.astype(np.float32)

def infer_and_draw_optflow(model, prev2_path, prev1_path, curr_path, device, orig_size=(640, 512), input_size=(256, 256)):
    img_prev2 = load_ir_image(prev2_path, input_size)
    img_prev1 = load_ir_image(prev1_path, input_size)
    img_current = load_ir_image(curr_path, input_size)

    flow1 = compute_flow(img_prev2, img_prev1)    
    flow2 = compute_flow(img_prev1, img_current)   

    img_current_norm = (img_current / 255.0).astype(np.float32)
    img_current_tensor = torch.from_numpy(img_current_norm).unsqueeze(0)  

    tensor_5ch = torch.cat([
        img_current_tensor,
        torch.from_numpy(flow1),
        torch.from_numpy(flow2)
    ], dim=0).unsqueeze(0).to(device) 
    model.eval()
    with torch.no_grad():
        logits, box_preds = model(tensor_5ch)
        bbox = box_preds[0].cpu().numpy() 
    x1 = int(bbox[0] * orig_size[0])
    y1 = int(bbox[1] * orig_size[1])
    x2 = int(bbox[2] * orig_size[0])
    y2 = int(bbox[3] * orig_size[1])

    img_pil_orig = Image.open(curr_path).convert("RGB")
    draw = ImageDraw.Draw(img_pil_orig)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

    display(img_pil_orig)
    return img_pil_orig, (x1, y1, x2, y2)

prev2_path = "/kaggle/input/valzip/val/images/1004.jpg"
prev1_path = "/kaggle/input/valzip/val/images/1005.jpg"
curr_path  = "/kaggle/input/valzip/val/images/1006.jpg"
model = EfficientNetDetectorOpticalFlow()

weights_path = "/kaggle/input/optical_flow_epoch_4/other/default/1/detector_model_optical_epoch_4.pth" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)     
    model.to(device)
    model.eval() 
    print(f"Model weights loaded successfully from {weights_path} to {device}.")

except FileNotFoundError:
    print(f"Error: The file '{weights_path}' was not found.")
except Exception as e:
    print(f"An error occurred while loading weights: {e}")

infer_and_draw_optflow(model, prev2_path, prev1_path, curr_path, device)
