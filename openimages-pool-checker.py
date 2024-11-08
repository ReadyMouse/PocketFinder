import os
import json
from PIL import Image
from tqdm import tqdm

class DatasetChecker:
    def __init__(self, dataset_dir="pool_table_dataset"):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.masks_dir = os.path.join(dataset_dir, "masks")
        self.annotations_dir = os.path.join(dataset_dir, "annotations")

    def check_dataset(self):
        """Check what was actually downloaded and create a report"""
        print("\nChecking dataset contents...")
        
        # Check directories exist
        dirs_exist = {
            "images": os.path.exists(self.images_dir),
            "masks": os.path.exists(self.masks_dir),
            "annotations": os.path.exists(self.annotations_dir)
        }
        
        print("\nDirectory Status:")
        for dir_name, exists in dirs_exist.items():
            print(f"{dir_name}: {'✓' if exists else '✗'}")

        # Count and validate files
        stats = {
            "total_images": 0,
            "valid_images": 0,
            "corrupted_images": [],
            "total_masks": 0,
            "valid_masks": 0,
            "corrupted_masks": [],
            "image_sizes": {},
            "mask_sizes": {}
        }

        # Check images
        if dirs_exist["images"]:
            print("\nChecking images...")
            image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            stats["total_images"] = len(image_files)
            
            for img_file in tqdm(image_files, desc="Validating images"):
                img_path = os.path.join(self.images_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        size = f"{img.width}x{img.height}"
                        stats["image_sizes"][size] = stats["image_sizes"].get(size, 0) + 1
                        stats["valid_images"] += 1
                except Exception as e:
                    stats["corrupted_images"].append((img_file, str(e)))

        # Check masks
        if dirs_exist["masks"]:
            print("\nChecking masks...")
            mask_files = [f for f in os.listdir(self.masks_dir) if f.endswith('.png')]
            stats["total_masks"] = len(mask_files)
            
            for mask_file in tqdm(mask_files, desc="Validating masks"):
                mask_path = os.path.join(self.masks_dir, mask_file)
                try:
                    with Image.open(mask_path) as mask:
                        size = f"{mask.width}x{mask.height}"
                        stats["mask_sizes"][size] = stats["mask_sizes"].get(size, 0) + 1
                        stats["valid_masks"] += 1
                except Exception as e:
                    stats["corrupted_masks"].append((mask_file, str(e)))

        # Check annotations
        annotations_found = []
        if dirs_exist["annotations"]:
            annotations_found = [f for f in os.listdir(self.annotations_dir) if f.endswith('.json')]

        # Create report
        report = {
            "directory_status": dirs_exist,
            "files_found": {
                "images": stats["total_images"],
                "masks": stats["total_masks"],
                "annotations": len(annotations_found)
            },
            "validation_results": {
                "valid_images": stats["valid_images"],
                "corrupted_images": len(stats["corrupted_images"]),
                "valid_masks": stats["valid_masks"],
                "corrupted_masks": len(stats["corrupted_masks"])
            },
            "image_size_distribution": stats["image_sizes"],
            "mask_size_distribution": stats["mask_sizes"],
            "corrupted_files": {
                "images": stats["corrupted_images"],
                "masks": stats["corrupted_masks"]
            }
        }

        # Save report
        report_path = os.path.join(self.dataset_dir, "dataset_validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\nDataset Validation Summary:")
        print(f"Images found: {stats['total_images']} (Valid: {stats['valid_images']})")
        print(f"Masks found: {stats['total_masks']} (Valid: {stats['valid_masks']})")
        print(f"Annotations found: {len(annotations_found)}")
        
        if stats["corrupted_images"]:
            print("\nCorrupted images found:")
            for img, error in stats["corrupted_images"][:5]:
                print(f"  - {img}: {error}")
            if len(stats["corrupted_images"]) > 5:
                print(f"  ... and {len(stats['corrupted_images']) - 5} more")
                
        print("\nImage size distribution:")
        for size, count in stats["image_sizes"].items():
            print(f"  {size}: {count} images")

        print(f"\nFull validation report saved to: {report_path}")

def main():
    try:
        checker = DatasetChecker()
        checker.check_dataset()
    except Exception as e:
        print(f"\nError during validation: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
