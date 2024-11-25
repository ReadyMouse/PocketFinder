import json
import os
import shutil
from PIL import Image
from tqdm import tqdm

class OpenImagesToYOLOClassification:
    def __init__(self, pool_dataset_dir="pool_table_dataset", 
                 negative_dataset_dir="negative_dataset",
                 output_dir="yolo_pool_classification"):
        self.pool_dataset_dir = pool_dataset_dir
        self.negative_dataset_dir = negative_dataset_dir
        self.output_dir = output_dir
        
        # Classification directory structure
        self.train_dir = os.path.join(output_dir, "train")
        self.val_dir = os.path.join(output_dir, "val")
        
        # Create class directories
        for split in [self.train_dir, self.val_dir]:
            os.makedirs(os.path.join(split, "pool_table"), exist_ok=True)
            os.makedirs(os.path.join(split, "no_pool_table"), exist_ok=True)
        
        # Create dataset.yaml
        self.create_yaml()

    def create_yaml(self):
        """Create YOLO classification dataset.yaml file"""
        yaml_content = f"""path: {os.path.abspath(self.output_dir)}
        train: train  # train images
        val: val      # val images

        # Classes
        names:
        - no_pool_table
        - pool_table

        # Number of classes
        nc: 2"""

        with open(os.path.join(self.output_dir, "dataset.yaml"), "w") as f:
            f.write(yaml_content)

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
                # Randomly assign to train or val (80/20 split)
                split_dir = self.train_dir if hash(sample['filename']) % 100 < 80 else self.val_dir
                
                # Copy image to pool_table directory
                src_img_path = os.path.join(self.pool_dataset_dir, "images", sample['filename'])
                dst_img_path = os.path.join(split_dir, "pool_table", sample['filename'])
                
                if not os.path.exists(src_img_path):
                    print(f"Warning: Source image not found: {src_img_path}")
                    continue
                
                shutil.copy2(src_img_path, dst_img_path)
            
            except Exception as e:
                print(f"Error processing pool table image {sample['filename']}: {str(e)}")

    def process_negative_dataset(self):
        """Process negative dataset (images without pool tables)"""
        print("\nProcessing negative dataset...")
        
        annotations_file = os.path.join(self.negative_dataset_dir, "annotations", "dataset_summary.json")
        if not os.path.exists(annotations_file):
            print(f"Warning: Could not find annotations file: {annotations_file}")
            return
        
        with open(annotations_file) as f:
            data = json.load(f)
        
        for sample in tqdm(data['samples'], desc="Converting negative images"):
            try:
                # Randomly assign to train or val (80/20 split)
                split_dir = self.train_dir if hash(sample['filename']) % 100 < 80 else self.val_dir
                
                # Copy image to no_pool_table directory
                src_img_path = os.path.join(self.negative_dataset_dir, "images", sample['filename'])
                dst_img_path = os.path.join(split_dir, "no_pool_table", sample['filename'])
                
                if not os.path.exists(src_img_path):
                    print(f"Warning: Source image not found: {src_img_path}")
                    continue
                
                shutil.copy2(src_img_path, dst_img_path)
            
            except Exception as e:
                print(f"Error processing negative image {sample['filename']}: {str(e)}")

    def convert(self):
        """Convert both datasets to YOLO classification format"""
        print("Starting conversion to YOLO classification format...")
        
        self.process_pool_dataset()
        self.process_negative_dataset()
        
        # Print summary
        train_pool = len(os.listdir(os.path.join(self.train_dir, "pool_table")))
        train_no_pool = len(os.listdir(os.path.join(self.train_dir, "no_pool_table")))
        val_pool = len(os.listdir(os.path.join(self.val_dir, "pool_table")))
        val_no_pool = len(os.listdir(os.path.join(self.val_dir, "no_pool_table")))
        
        print("\nConversion completed!")
        print(f"Training set:")
        print(f"  Pool table images: {train_pool}")
        print(f"  No pool table images: {train_no_pool}")
        print(f"Validation set:")
        print(f"  Pool table images: {val_pool}")
        print(f"  No pool table images: {val_no_pool}")
        print(f"\nDataset saved in: {self.output_dir}")
        print(f"YAML file created: {os.path.join(self.output_dir, 'dataset.yaml')}")

def main():
    try:
        converter = OpenImagesToYOLOClassification()
        converter.convert()
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()