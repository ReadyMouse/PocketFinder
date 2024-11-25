from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import argparse
import json

def run_inference(model_path, image_path, output_dir, conf_threshold=0.5):
    """
    Run classification inference on a single image or directory of images
    """
    # Load the model
    model = YOLO(model_path)
    print('Successfully loaded the model weights.')
    
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
            save=True,   # Save the results
            project=output_dir,
            name='predictions'
        )

        # Extract classification results
        result = {
            'class_name': results[0].names[results[0].probs.top1],  # Get class name
            'confidence': float(results[0].probs.top1conf),  # Get confidence
            'class_index': int(results[0].probs.top1)  # Get class index
        }
            
        all_results[os.path.basename(img_path)] = result
        
        print(f"\nProcessed {img_path}")
        print(f"Prediction: {result['class_name']} ({result['confidence']:.2f} confidence)")
    
    # Save results to JSON
    os.makedirs(output_dir / 'predictions', exist_ok=True)
    results_file = output_dir / 'predictions' / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"- JSON results: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference on images')
    #parser.add_argument('--model', type=str, required=True, help='Path to the model weights file')
    #parser.add_argument('--source', type=str, required=True, help='Path to image or directory of images')
    #parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (0-1)')
    
    #args = parser.parse_args()

    # model_path = './Users/mouse/src/PocketFinder/yolov8x-seg.pt' # base yolo 
    model_path = '/Users/mouse/src/PocketFinder/First-working-copy/weights/best.pt'
    image_path = '/Users/mouse/src/PocketFinder/my_own_pics/toad_hall_sample1.jpeg'
    # image_path = '/Users/mouse/src/Pocket-Finder/my_own_pics/no_pool_table.jpeg'
    save_path = Path('/Users/mouse/src/PocketFinder/my_own_pics_results')

    run_inference(model_path, image_path , save_path, 0.5)

if __name__ == "__main__":
    main()