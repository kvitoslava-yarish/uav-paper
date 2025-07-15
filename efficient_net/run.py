import torch
from torchvision import transforms as T
from PIL import Image, ImageDraw
from model import EfficientNetDetector

def infer_and_draw_ir(model, image_path, device, orig_size=(640, 512), input_size=(256, 256)):
    img_pil = Image.open(image_path).convert("L")  
    img_resized = img_pil.resize(input_size)
    img_tensor = T.ToTensor()(img_resized) 
    img_tensor = img_tensor.repeat(3, 1, 1)  
    img_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, box_preds = model(img_tensor)
        bbox = box_preds[0].cpu().numpy()

    x1 = int(bbox[0] * orig_size[0])
    y1 = int(bbox[1] * orig_size[1])
    x2 = int(bbox[2] * orig_size[0])
    y2 = int(bbox[3] * orig_size[1])

    img_draw = img_pil.convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

    from IPython.display import display
    display(img_draw)

    return img_draw, (x1, y1, x2, y2)

model = EfficientNetDetector()

weights_path = "/kaggle/input/efficientnet_baseline_epoch_3/other/default/1/detector_model_epoch_3.pth" 
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
image_path = "/kaggle/input/valzip/val/images/10008.jpg"  
infer_and_draw_ir(model, image_path, device)
