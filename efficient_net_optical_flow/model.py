
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

class FlatInfraredDatasetOpticalFlow(Dataset):
    def __init__(self, json_path, image_dir, transform=None, target_size=(256, 256)):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.exists = data["exist"]
        self.bboxes = data["gt_rect"]
        self.image_dir = image_dir

        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform

        self.image_files = sorted(os.listdir(image_dir), key=natural_sort_key)

        self.orig_width = 640
        self.orig_height = 512
        self.target_width, self.target_height = target_size

        self.valid_data_indices = []
        for i in range(len(self.image_files)):
            if self.exists[i] == 1:
                if i >= 2:
                    self.valid_data_indices.append(i)

        print(f"Found {len(self.valid_data_indices)} valid sequences for detection with optical flow.")

    def __len__(self):
        return len(self.valid_data_indices)

    def __getitem__(self, idx):
        frame_n_overall_idx = self.valid_data_indices[idx]

        filename_n = self.image_files[frame_n_overall_idx]
        filename_n_minus_1 = self.image_files[frame_n_overall_idx - 1]
        filename_n_minus_2 = self.image_files[frame_n_overall_idx - 2]

        bbox = self.bboxes[frame_n_overall_idx]
        x, y, w, h = bbox

        img_n_minus_2_pil = Image.open(os.path.join(self.image_dir, filename_n_minus_2)).convert('L')
        img_n_minus_1_pil = Image.open(os.path.join(self.image_dir, filename_n_minus_1)).convert('L')
        img_n_pil = Image.open(os.path.join(self.image_dir, filename_n)).convert('L')

        size_tuple = (self.target_width, self.target_height)
        img_n_minus_2_resized_pil = img_n_minus_2_pil.resize(size_tuple)
        img_n_minus_1_resized_pil = img_n_minus_1_pil.resize(size_tuple)
        img_n_resized_pil = img_n_pil.resize(size_tuple)

        frame_0_np = np.array(img_n_minus_2_resized_pil)
        frame_1_np = np.array(img_n_minus_1_resized_pil)
        frame_2_np = np.array(img_n_resized_pil)

        flow_01 = cv2.calcOpticalFlowFarneback(frame_0_np, frame_1_np, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_12 = cv2.calcOpticalFlowFarneback(frame_1_np, frame_2_np, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        max_flow_val = 20.0

        flow_01_tensor = torch.from_numpy(flow_01).permute(2, 0, 1).float()
        flow_12_tensor = torch.from_numpy(flow_12).permute(2, 0, 1).float()

        flow_01_tensor = torch.clamp(flow_01_tensor / max_flow_val, -1.0, 1.0)
        flow_12_tensor = torch.clamp(flow_12_tensor / max_flow_val, -1.0, 1.0)

        img_n_tensor = self.transform(img_n_resized_pil)

        combined_input_tensor = torch.cat([img_n_tensor, flow_01_tensor, flow_12_tensor], dim=0)

        x1_scaled = x * self.target_width / self.orig_width
        y1_scaled = y * self.target_height / self.orig_height
        x2_scaled = (x + w) * self.target_width / self.orig_width
        y2_scaled = (y + h) * self.target_height / self.orig_height

        x1_norm = max(0.0, min(1.0, x1_scaled / self.target_width))
        y1_norm = max(0.0, min(1.0, y1_scaled / self.target_height))
        x2_norm = max(0.0, min(1.0, x2_scaled / self.target_width))
        y2_norm = max(0.0, min(1.0, y2_scaled / self.target_height))

        target = {
            "boxes": torch.tensor([[x1_norm, y1_norm, x2_norm, y2_norm]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64)
        }

        return combined_input_tensor, target


weights_path = "detector_model_optical_epoch_4.pth"
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
