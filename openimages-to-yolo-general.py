import json
import os
import shutil
from PIL import Image
from tqdm import tqdm

class OpenImagesToYOLO:
    def __init__(self, pool_dataset_dir="pool_table_dataset", 
                 negative_dataset_dir="negative_dataset",
                 output_dir="yolo_pool_dataset"):
        self.pool_dataset_dir = pool_dataset_dir
        self.negative_dataset_dir = negative_dataset_dir
        self.output_dir = output_dir
        
        # YOLO directory structure
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Class mapping (for YOLO format)
        self.class_map = {
            "Billiard table": 0  # Pool table is class 0
        }
        
        # Create dataset.yaml
        self.create_yaml()

    def create_yaml(self):
        """Create YOLO dataset.yaml file"""
        yaml_content = f"""path: {os.path.abspath(self.output_dir)}
train: images  # train images relative to 'path'
val: images  # val images relative to 'path'

# Classes
names:
  0: pool_table

# Number of classes
nc: 1"""

        with open(os.path.join(self.output_dir, "dataset.yaml"), "w") as f:
            f.write(yaml_content)

    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """Convert OpenImages bbox to YOLO format"""
        # OpenImages format: [x_min, y_min, x_max, y_max] in absolute pixels
        # YOLO format: [x_center, y_center, width, height] normalized to [0, 1]
        
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to normalized coordinates
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Ensure values are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return [x_center, y_center, width, height]

    def process_pool_dataset(self):
        """Process pool table dataset"""
        print("\nProcessing pool table dataset...")
        
        # Load annotations
        annotations_file = os.path.join(self.pool_dataset_dir, "annotations", "dataset_summary.json")
        if not os.path.exists(annotations_file):
            print(f"Warning: Could not find annotations file: {annotations_file}")
            return
        
        with open(annotations_file) as f:
            data = json.load(f)
        
        # Process each image
        for sample in tqdm(data['samples'], desc="Converting pool table images"):
            try:
                # Copy image
                src_img_path = os.path.join(self.pool_dataset_dir, "images", sample['filename'])
                dst_img_path = os.path.join(self.images_dir, sample['filename'])
                
                if not os.path.exists(src_img_path):
                    print(f"Warning: Source image not found: {src_img_path}")
                    continue
                
                shutil.copy2(src_img_path, dst_img_path)
                
                # Create YOLO label file
                label_filename = os.path.splitext(sample['filename'])[0] + '.txt'
                label_path = os.path.join(self.labels_dir, label_filename)
                
                # Get image dimensions
                with Image.open(src_img_path) as img:
                    img_width, img_height = img.size
                
                # Write YOLO format labels
                with open(label_path, 'w') as f:
                    class_id = self.class_map["Billiard table"]
                    if 'masks' in sample:  # If we have detection data
                        for mask in sample['masks']:
                            if 'bbox' in mask:  # If we have bounding box data
                                yolo_bbox = self.convert_bbox_to_yolo(
                                    mask['bbox'], img_width, img_height)
                                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
            
            except Exception as e:
                print(f"Error processing pool table image {sample['filename']}: {str(e)}")

    def process_negative_dataset(self):
        """Process negative dataset (images without pool tables)"""
        print("\nProcessing negative dataset...")
        
        # Load annotations
        annotations_file = os.path.join(self.negative_dataset_dir, "annotations", "dataset_summary.json")
        if not os.path.exists(annotations_file):
            print(f"Warning: Could not find annotations file: {annotations_file}")
            return
        
        with open(annotations_file) as f:
            data = json.load(f)
        
        # Process each image
        for sample in tqdm(data['samples'], desc="Converting negative images"):
            try:
                # Copy image
                src_img_path = os.path.join(self.negative_dataset_dir, "images", sample['filename'])
                dst_img_path = os.path.join(self.images_dir, sample['filename'])
                
                if not os.path.exists(src_img_path):
                    print(f"Warning: Source image not found: {src_img_path}")
                    continue
                
                shutil.copy2(src_img_path, dst_img_path)
                
                # Create empty label file (no pool tables)
                label_filename = os.path.splitext(sample['filename'])[0] + '.txt'
                label_path = os.path.join(self.labels_dir, label_filename)
                
                # Create empty label file
                open(label_path, 'w').close()
            
            except Exception as e:
                print(f"Error processing negative image {sample['filename']}: {str(e)}")

    def convert(self):
        """Convert both datasets to YOLO format"""
        print("Starting conversion to YOLO format...")
        
        self.process_pool_dataset()
        self.process_negative_dataset()
        
        # Print summary
        num_images = len(os.listdir(self.images_dir))
        num_labels = len(os.listdir(self.labels_dir))
        
        print("\nConversion completed!")
        print(f"Total images converted: {num_images}")
        print(f"Total label files created: {num_labels}")
        print(f"\nDataset saved in: {self.output_dir}")
        print(f"YAML file created: {os.path.join(self.output_dir, 'dataset.yaml')}")

def main():
    try:
        converter = OpenImagesToYOLO()
        converter.convert()
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
