#!/usr/bin/env python3
import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def load_model(model_path):
    """Load the model from the given path"""
    if model_path == "pretrained":
        # Use default pretrained model if no specific model provided
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    else:
        # Load custom trained model
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
        model.load_state_dict(torch.load(model_path))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model

def ensure_dir(file_path):
    """Ensure directory exists for the given file path"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def run_inference(image_dir, model_path, output_predictions):
    """Run zero-shot inference on all images in the directory"""
    # Define the text labels we want to detect (adjust as needed)
    text_labels = "car, person, bicycle, motorcycle, bus, truck, traffic light, stop sign"
    
    # Load model
    processor, model = load_model(model_path)
    
    # Prepare output structure
    output = {
        "images": [],
        "annotations": []
    }
    
    # Process each image in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    annotation_id = 1
    
    for image_id, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        
        # Add image to output
        output["images"].append({
            "id": image_id,
            "file_name": image_file
        })
        
        # Run inference
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )[0]
        
        # Convert results to output format
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            xmin, ymin, xmax, ymax = box.tolist()
            width = xmax - xmin
            height = ymax - ymin
            
            output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": label.item() + 1,  # Assuming 1-based category IDs
                "bbox": [xmin, ymin, width, height],
                "score": score.item()
            })
            annotation_id += 1
    
    # Ensure output directory exists
    ensure_dir(output_predictions)
    
    # Save predictions
    with open(output_predictions, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Path to directory containing input images")
    parser.add_argument("model_path", help="Path to pretrained or fine-tuned model")
    parser.add_argument("output_predictions", help="Path to save output predictions JSON file")
    
    args = parser.parse_args()
    
    run_inference(args.image_dir, args.model_path, args.output_predictions)
