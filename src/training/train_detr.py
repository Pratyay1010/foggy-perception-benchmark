
import os
import json
import argparse
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Configuration
BATCH_SIZE = 2
NUM_EPOCHS = 20
LR = 1e-5
WEIGHT_DECAY = 1e-4
GRADIENT_ACCUMULATION_STEPS = 4
THRESHOLD = 0.3

# Class mapping
COCO_TO_CUSTOM = {
    1: 0, 3: 1, 7: 2, 8: 3, 4: 4, 2: 5, 6: 6
}
CUSTOM_CLASSES = ['person', 'car', 'train', 'truck', 'motorcycle', 'bicycle', 'bus']

def get_transform():
    return T.Compose([
        T.Resize((400, 600)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class FoggyDataset(Dataset):
    def __init__(self, root, annotation_path, transform=None):
        self.root = root
        self.transform = transform if transform is not None else get_transform()
        with open(annotation_path) as f:
            data = json.load(f)
        self.images = {img['id']: img for img in data['images']}
        self.annotations = defaultdict(list)
        for ann in data['annotations']:
            orig_id = ann['category_id']
            if data['categories'][orig_id-1]['name'] == 'rider':
                ann['category_id'] = 1
            if orig_id in COCO_TO_CUSTOM:
                ann['category_id'] = COCO_TO_CUSTOM[orig_id]
                self.annotations[ann['image_id']].append(ann)
        
        # Filter out images with no annotations
        self.valid_image_ids = [img_id for img_id in self.images 
                              if len(self.annotations[img_id]) > 0]

    def __getitem__(self, idx):
        img_id = self.valid_image_ids[idx]
        img_info = self.images[img_id]
        img = Image.open(os.path.join(self.root, img_info['file_name'])).convert('RGB')
        boxes, labels = [], []
        
        for ann in self.annotations[img_id]:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
            
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'class_labels': torch.tensor(labels, dtype=torch.long),
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([img_info['height'], img_info['width']])
        }
        
        return self.transform(img), target

    def __len__(self):
        return len(self.valid_image_ids)

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(images), targets

def evaluate(model, dataloader, processor, device, output_file=None):
    # Get the correct annotation path
    annotation_dir = os.path.dirname(dataloader.dataset.root)
    annotation_path = os.path.join(annotation_dir, "annotations/instances_val.json")
    
    coco = COCO(annotation_path)
    results = []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            outputs = model(images.to(device))
            processed = processor.post_process_object_detection(
                outputs, threshold=THRESHOLD,
                target_sizes=[t['orig_size'] for t in targets]
            )
            for i, (target, pred) in enumerate(zip(targets, processed)):
                image_id = target['image_id'].item()
                for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        'image_id': image_id,
                        'category_id': label.item() + 1,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': score.item()
                    })
    
    if output_file:
        # Save predictions in the required format
        predictions = {
            "images": [{"id": img_id, "file_name": f"{img_id:05d}.jpg"} for img_id in coco.getImgIds()],
            "annotations": results
        }
        with open(output_file, 'w') as f:
            json.dump(predictions, f)
    
    if len(results) > 0:
        coco_dt = coco.loadRes(results)
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0]
    else:
        print("No predictions to evaluate!")
        return 0.0

# [Rest of your code remains the same...]

def eval_pretrained(args):
    # Initialize dataset and dataloader
    val_set = FoggyDataset(args.image_dir, os.path.join(os.path.dirname(args.image_dir), "annotations/instances_val.json"))
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Load pretrained model
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Run evaluation and save predictions
    mAP = evaluate(model, val_loader, processor, device, args.output_pred)
    print(f"\nMean Average Precision (mAP@0.5:0.95): {mAP:.4f}")

def eval_finetuned(args):
    # Initialize dataset and dataloader
    val_set = FoggyDataset(args.image_dir, os.path.join(os.path.dirname(args.image_dir), "annotations/instances_val.json"))
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Load fine-tuned model
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr", num_labels=len(CUSTOM_CLASSES))
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    
    # Load state dict
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict['model_state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Run evaluation and save predictions
    mAP = evaluate(model, val_loader, processor, device, args.output_pred)
    print(f"\nMean Average Precision (mAP@0.5:0.95): {mAP:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Deformable DETR for Foggy Object Detection")
    
    # Main mode selection
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "eval_pretrained", "eval_finetuned"],
                       help="Operation mode: train or eval")
    
    # Training arguments
    parser.add_argument("--strategy", type=int, choices=[1, 2, 3], default=1,
                       help="Fine-tuning strategy (1: full, 2: decoder-only, 3: encoder-only)")
    parser.add_argument("--data_root", type=str,
                       help="Root directory of the dataset (for training)")
    parser.add_argument("--output", type=str,
                       help="Path to save trained model (for training)")
    
    # Evaluation arguments
    parser.add_argument("--image_dir", type=str,
                       help="Directory containing images for evaluation")
    parser.add_argument("--model_path", type=str,
                       help="Path to model weights for evaluation")
    
    # Output file
    parser.add_argument("--output_pred", type=str,
                       help="Path to save predictions (for evaluation)")
    
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "eval_pretrained":
        eval_pretrained(args)
    elif args.mode == "eval_finetuned":
        eval_finetuned(args)

if __name__ == "__main__":
    main()
