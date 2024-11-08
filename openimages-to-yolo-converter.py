import os
import shutil
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class OpenImagesToYOLO:
    def __init__(self, input_dir="pool_table_dataset", output_dir="yolo_pool_dataset"):
        # Input directories
        self.input_dir = input_dir
        self.input_images = os.path.join(input_dir, "images")
        self.input_masks = os.path.join(input_dir, "masks")
        self.input_annotations = os.path.join(input_dir, "annotations")
        
        # Output directories
        self.output_dir = output_dir
        self.output_images = os.path.join(output_dir, "images")
        self.output_labels = os.path.join(output_dir, "labels")
        
        # Create YOLO directory structure
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.output_images, split), exist_ok=True)
            os.makedirs(os.path.join(self.output_labels, split), exist_ok=True)
        
        # Pool table is our only class
        self.class_name = "pool_table"
        self.class_id = 0

    def mask_to_polygons(self, mask_path, image_width, image_height):
        """Convert binary mask to YOLO polygon format with proper normalization"""
        try:
            # Read mask
            mask = np.array(Image.open(mask_path))
            
            # Find contours using numpy operations
            from skimage import measure
            contours = measure.find_contours(mask, 0.5)
            
            # Convert to YOLO format (normalized coordinates)
            yolo_polygons = []
            for contour in contours:
                # Reduce number of points to make the annotation more manageable
                if len(contour) > 100:
                    indices = np.linspace(0, len(contour) - 1, 100, dtype=int)
                    contour = contour[indices]
                
                # Normalize coordinates and clip to [0, 1] range
                polygon = []
                valid_polygon = True
                
                for point in contour:
                    # Convert from row,col to x,y and normalize
                    x = point[1] / image_width
                    y = point[0] / image_height
                    
                    # Clip coordinates to [0, 1] range
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))
                    
                    polygon.extend([x, y])
                
                if len(polygon) >= 6:  # YOLO needs at least 3 points
                    # Verify all coordinates are within bounds
                    if all(0 <= coord <= 1 for coord in polygon):
                        yolo_polygons.append(polygon)
            
            return yolo_polygons
            
        except Exception as e:
            print(f"Error processing mask {mask_path}: {str(e)}")
            return []

    def create_yolo_label(self, image_id, image_path, mask_paths):
        """Create YOLO format label file for an image"""
        try:
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Process all masks for this image
            all_polygons = []
            for mask_path in mask_paths:
                polygons = self.mask_to_polygons(mask_path, width, height)
                all_polygons.extend(polygons)
            
            return all_polygons
            
        except Exception as e:
            print(f"Error creating label for {image_id}: {str(e)}")
            return []

    def convert_dataset(self, val_split=0.2):
        """Convert the entire dataset to YOLO format"""
        print("Converting OpenImages pool table dataset to YOLO format...")
        
        # Load dataset summary if available
        summary_path = os.path.join(self.input_annotations, "dataset_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                dataset_info = json.load(f)
        else:
            print("No dataset summary found. Scanning directories...")
            dataset_info = self.scan_dataset()
        
        # Get all image IDs and their corresponding masks
        image_mask_pairs = []
        for sample in dataset_info.get('samples', []):
            image_file = sample['filename']
            mask_files = [mask['mask_file'] for mask in sample.get('masks', [])]
            if mask_files:  # Only include images that have masks
                image_mask_pairs.append((image_file, mask_files))
        
        # Split into train/val
        train_pairs, val_pairs = train_test_split(
            image_mask_pairs, test_size=val_split, random_state=42
        )
        
        # Process splits
        splits = {
            'train': train_pairs,
            'val': val_pairs
        }
        
        # Convert each split
        for split_name, pairs in splits.items():
            print(f"\nProcessing {split_name} split...")
            for image_file, mask_files in tqdm(pairs):
                try:
                    # Source paths
                    image_path = os.path.join(self.input_images, image_file)
                    mask_paths = [os.path.join(self.input_masks, mf) for mf in mask_files]
                    
                    # Destination paths
                    dest_image = os.path.join(self.output_images, split_name, image_file)
                    label_file = os.path.join(
                        self.output_labels, 
                        split_name, 
                        os.path.splitext(image_file)[0] + '.txt'
                    )
                    
                    # Copy image
                    shutil.copy2(image_path, dest_image)
                    
                    # Create YOLO label
                    polygons = self.create_yolo_label(image_file, image_path, mask_paths)
                    
                    # Save label file
                    with open(label_file, 'w') as f:
                        for polygon in polygons:
                            # YOLO format: class_id x1 y1 x2 y2 ...
                            points_str = ' '.join([f"{p:.6f}" for p in polygon])
                            f.write(f"{self.class_id} {points_str}\n")
                            
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
                    continue
        
        # Create dataset YAML file
        yaml_content = {
            'path': os.path.abspath(self.output_dir),
            'train': os.path.join('images', 'train'),
            'val': os.path.join('images', 'val'),
            'names': {
                self.class_id: self.class_name
            }
        }
        
        with open(os.path.join(self.output_dir, 'dataset.yaml'), 'w') as f:
            import yaml
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        # Print summary
        print("\nDataset conversion completed!")
        print(f"Training samples: {len(train_pairs)}")
        print(f"Validation samples: {len(val_pairs)}")
        print(f"\nDataset saved in: {self.output_dir}")
        print("Directory structure:")
        print(f"  {self.output_dir}/")
        print(f"  ├── images/")
        print(f"  │   ├── train/")
        print(f"  │   └── val/")
        print(f"  ├── labels/")
        print(f"  │   ├── train/")
        print(f"  │   └── val/")
        print(f"  └── dataset.yaml")

def main():
    try:
        converter = OpenImagesToYOLO()
        converter.convert_dataset()
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
