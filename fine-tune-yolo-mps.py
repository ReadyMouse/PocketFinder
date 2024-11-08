from ultralytics import YOLO
import os
import yaml
import torch
from pathlib import Path
import shutil
import argparse

class PoolTableTrainer:
    def __init__(self, data_yaml="yolo_pool_dataset/dataset.yaml",
                 epochs=100,
                 batch_size=16,
                 imgsz=640,
                 device=None):
        """
        Initialize the pool table trainer
        """
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        
        # Device selection with MPS support
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Create runs directory for outputs
        self.runs_dir = Path('runs/segment')
        os.makedirs(self.runs_dir, exist_ok=True)
        
        # Load dataset config
        with open(data_yaml, 'r') as f:
            self.dataset_config = yaml.safe_load(f)

    def setup_model(self, pretrained_weights="yolov8s-seg.pt"):
        """
        Setup YOLOv8 model with pretrained weights
        """
        print(f"\nSetting up YOLOv8 model with {pretrained_weights}")
        print(f"Using device: {self.device}")
        
        try:
            # Download pretrained weights if not exists
            if not os.path.exists(pretrained_weights):
                print(f"Downloading {pretrained_weights}...")
                model = YOLO(pretrained_weights)
            else:
                model = YOLO(pretrained_weights)
            
            print(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            print(f"Error setting up model: {str(e)}")
            raise

    def train(self, model):
        """
        Train the model with optimized settings
        """
        print("\nStarting training...")
        print(f"Training configuration:")
        print(f"- Epochs: {self.epochs}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Image size: {self.imgsz}")
        print(f"- Device: {self.device}")
        
        try:
            # Start training with optimized parameters
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.imgsz,
                device=self.device,
                cache='disk',  # More stable than RAM cache
                workers=4,     # Adjust based on your CPU
                patience=20,   # Early stopping patience
                save=True,     # Save checkpoints
                save_period=10,# Save every 10 epochs
                pretrained=True,
                optimizer='Adam',
                lr0=0.001,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                plots=True,
                exist_ok=True,
                overlap_mask=True,
                mask_ratio=4,
                single_cls=True,  # Since we only have one class
                rect=True,
                amp=True,
                close_mosaic=10
            )
            
            return results
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def validate(self, model):
        """
        Validate the model
        """
        print("\nRunning validation...")
        try:
            results = model.val(
                data=self.data_yaml,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device=self.device,
                split='val',
                plots=True
            )
            return results
        
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            raise

    def export_model(self, model, format='onnx'):
        """
        Export the model to specified format
        """
        print(f"\nExporting model to {format}...")
        try:
            model.export(format=format)
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
            raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv8 for pool table segmentation')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--weights', type=str, default='yolov8s-seg.pt', help='initial weights path')
    parser.add_argument('--device', type=str, default=None, help='device (mps, cuda, or cpu)')
    args = parser.parse_args()

    try:
        print("PyTorch device availability:")
        print(f"MPS (Apple Silicon): {torch.backends.mps.is_available()}")
        print(f"CUDA (NVIDIA): {torch.cuda.is_available()}")
        
        # Initialize trainer
        trainer = PoolTableTrainer(
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.img_size,
            device=args.device
        )
        
        # Setup model
        model = trainer.setup_model(args.weights)
        
        # Train model
        results = trainer.train(model)
        
        # Validate model
        val_results = trainer.validate(model)
        
        # Export model
        trainer.export_model(model)
        
        print("\nTraining completed successfully!")
        print(f"Results saved in: {trainer.runs_dir}")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
