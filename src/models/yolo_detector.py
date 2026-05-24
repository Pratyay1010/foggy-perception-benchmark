
import os
import json
import shutil
import argparse
import yaml
from tqdm import tqdm
from ultralytics import YOLO  # Ensure yolov8 is installed: pip install ultralytics


def convert_coco_to_yolo(json_path, output_label_dir):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img for img in coco["images"]}
    cat_to_class = {cat["id"]: i for i, cat in enumerate(coco["categories"])}
    class_names = [cat["name"] for cat in coco["categories"]]

    annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        annotations.setdefault(img_id, []).append(ann)

    for img_id, anns in tqdm(annotations.items(), desc=f"Processing {os.path.basename(json_path)}"):
        img_info = id_to_filename[img_id]
        width, height = img_info["width"], img_info["height"]

        file_name = os.path.basename(img_info["file_name"])
        file_stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(output_label_dir, f"{file_stem}.txt")

        with open(label_path, "w") as f:
            for ann in anns:
                cat = cat_to_class[ann["category_id"]]
                x, y, w, h = ann["bbox"]
                x_c = (x + w / 2) / width
                y_c = (y + h / 2) / height
                w /= width
                h /= height
                f.write(f"{cat} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    return class_names


def create_yaml_file(output_dir, class_names):
    yaml_content = {
        "train": os.path.join(output_dir, "train/images"),
        "val": os.path.join(output_dir, "val/images"),
        "nc": len(class_names),
        "names": class_names
    }

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
    return yaml_path


def prepare_dataset(input_dir, output_dir):
    splits = ["train", "val"]
    annotation_dir = os.path.join(input_dir, "annotations")

    for split in splits:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

        input_img_dir = os.path.join(input_dir, split)
        output_img_dir = os.path.join(output_dir, split, "images")
        label_dir = os.path.join(output_dir, split, "labels")

        for root, _, files in os.walk(input_img_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    shutil.copy2(os.path.join(root, file), os.path.join(output_img_dir, file))

        json_path = os.path.join(annotation_dir, f"instances_{split}.json")
        class_names = convert_coco_to_yolo(json_path, label_dir)

    return create_yaml_file(output_dir, class_names)


def train_yolo(dataset_root, save_path):
    output_dir = os.path.join(save_path, "yolo_dataset")
    yaml_path = prepare_dataset(dataset_root, output_dir)
    model = YOLO("yolo11x.pt")
    model.train(data=yaml_path,
                epochs=25,
                imgsz=640,
                batch=2,
                project=save_path,
                name="train",
                save=True)


def predict_yolo(image_dir, model_path, output_json):
    model = YOLO(model_path)
    results = model.predict(source=image_dir, save=False, stream=True)
    output = {"images": [], "annotations": [], "categories": []}
    for cid, name in model.names.items():
        output['categories'].append({"id": cid + 1, "name": name})
    ann_id = 1
    img_id = 1
    for res in results:
        fname = os.path.basename(res.path)
        output['images'].append({"id": img_id, "file_name": fname})
        boxes = res.boxes.data.cpu().numpy()
        for *xyxy, conf, cls in boxes:
            x1, y1, x2, y2 = xyxy
            output['annotations'].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls) + 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(conf)
            })
            ann_id += 1
        img_id += 1
    with open(output_json, 'w') as jf:
        json.dump(output, jf, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    if args.train:
        train_yolo(args.dataset_root, args.save_path)
    elif args.predict:
        predict_yolo(args.image_dir, args.model_path, args.output)
