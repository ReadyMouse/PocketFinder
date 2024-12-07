from ultralytics import YOLO
import os
import yaml
import torch
from pathlib import Path
import shutil
import argparse
from config import PATHS

class PoolTableTrainer:
    def __init__(self, data_yaml="dataset.yaml",
                 epochs=50,# was 100
                 batch_size=16,
                 imgsz=640,
                 device=None, 
                 runs_dir='',
                 pretrained_weights="yolov8n-cls.pt" ):
        """
        Initialize the pool table trainer
        """
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.runs_dir = runs_dir
        
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
        os.makedirs(self.runs_dir, exist_ok=True)
        
        # Load dataset config
        with open(data_yaml, 'r') as f:
            self.dataset_config = yaml.safe_load(f)

        # Download pretrained weights if not exists
        print(f"\nSetting up YOLOv8 model with {pretrained_weights}")
        print(f"Using device: {self.device}")

        if not os.path.exists(pretrained_weights):
            print(f"Downloading {pretrained_weights}...")
            self.model = YOLO(pretrained_weights)
        else:
            self.model = YOLO(pretrained_weights)
        print(f"Model loaded successfully on {self.device}")

        return 

    def train(self):
        """
        Train the model with optimized settings
        """
        print("\nStarting training...")
        print(f"Training configuration:")
        print(f"- Epochs: {self.epochs}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Image size: {self.imgsz}")
        print(f"- Device: {self.device}")
        model = self.model
    
        results = model.train(
            project=os.path.dirname(self.runs_dir),  # Parent directory
            name=os.path.basename(self.runs_dir),    # Run name
            data=os.path.dirname(self.data_yaml),    # path to yaml file
            epochs=self.epochs,                      # Number of epochs
            batch=self.batch_size,                   # Batch size
            imgsz=self.imgsz,                        # Image size
            device=self.device,                      # CPU/GPU
        
            # Classification-specific parameters
            pretrained=True,
            optimizer='Adam',
            lr0=0.001,  # Reduced learning rate for classification
            lrf=0.01,
            warmup_epochs=3,
            
            # Data handling
            workers=4,
            cache='ram',
            
            # Augmentation
            augment=True,
            mixup=0.1
            )

        return results

    def validate(self):
        """
        Validate the model
        """
        model = self.model
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

    def export_model(self, format='onnx'):
        """
        Export the model to specified format
        """
        model = self.model
        print(f"\nExporting model to {format}...")
        try:
            model.export(format=format)
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
            raise

    def inference(self, image_path, output_dir, conf_threshold=0.25):
        """
        Run inference on a single image or directory of images
        """        
        model = self.model
        # Handle both single images and directories
        image_paths = []
        if os.path.isfile(image_path):
            image_paths = [image_path]
        else:
            image_paths = [str(p) for p in Path(image_path).glob('*') 
                        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        all_results = {}

        for img_path in image_paths:
            # Run inference
            results = model.predict(
                source=img_path,
                conf=conf_threshold,
                show=True,  # Display the annotated image
                save=True,   # Save the results
                project=output_dir,
                name='predictions'
            )

            # Extract classification results
            result = {
                'class': results[0].probs.top1,  # Top class index
                'confidence': float(results[0].probs.top1conf),  # Confidence score
                'class_name': results[0].names[results[0].probs.top1]  # Class name
            }
            
            all_results[os.path.basename(img_path)] = result
            
            print(f"Processed {img_path}")
            print(f"Prediction: {result['class_name']} ({result['confidence']:.2f})")
        
        
        # Save results to JSON
        results_file = output_dir / 'predictions' / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"- Annotated images: {output_dir}/predictions/")
        print(f"- JSON results: {results_file}")

        return all_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv8 for pool table segmentation')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--weights', type=str, default='yolov8n-cls.pt', help='initial weights path')
    parser.add_argument('--device', type=str, default=None, help='device (mps, cuda, or cpu)')
    args = parser.parse_args()

    print("PyTorch device availability:")
    print(f"MPS (Apple Silicon): {torch.backends.mps.is_available()}")
    print(f"CUDA (NVIDIA): {torch.cuda.is_available()}")

    # Update runs_dir path construction
    project_dir = PATHS['project_dir']

    name = 'run/epoch10'
    runs_dir = os.path.join(project_dir, name)

    name2='yolo_pool_classification/dataset.yaml'
    data_yaml = os.path.join(project_dir, name2)
    
    # Initialize trainer
    trainer = PoolTableTrainer(
        epochs=args.epochs,
        data_yaml=data_yaml,
        batch_size=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        runs_dir=runs_dir 
    )

    trainer.inspect_training_weights("start")
    
    # Train model
    results = trainer.train()
    
    # Validate model
    val_results = trainer.validate()
    
    # Export model
    trainer.export_model()
    
    print("\nTraining completed successfully!")
    print(f"Results saved in: {trainer.runs_dir}")        

if __name__ == "__main__":
    main()
