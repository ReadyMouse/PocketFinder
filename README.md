# PocketFinder
Automated Finder for Pocket billiards venues. 

Challenges: 
1. Ambiguity in text
2. Limited pixels on object

Finding bars, resturants, and hotels with pool tables is particuarlly hard problem because these places don't often list billiards or pool tables as an amentiy on their websites. Sometimes Google reviewers will include a note in their feedback, however there is high ambiguity between "pool tables", "pool", and "table". Eg "I really enjoyed playing in the pool, and got a table right away at the resturant." Rarely will a reviewer use "billiards" for a dive bar, unless it's a pool hall or higher-scale establishment. Scanning the photos, pool tables often show as small slivers in the back of full-room shots, making identifying the billiards table difficult. 

This repo approaches this problem using an image segmentation approach of a pre-trained YOLOv8, fine-tuned on pool tables from OpenImages. 

## Step 0: 
Set up a working environmnet based off the requirements.txt

## Step 1: Download the pool table images from openImages 

```bash
python3 openimages-pool-table-image-downloader.py 
```

## Step 2: (Optional) Check the images downloaded properly 

```bash 
python3 openimages-pool-checker.py 
```

## Step 3: Convert Openimages to YOLO format 

```bash
python3 openimages-to-yolo-converter.py
```

## Step 4: YOLO fine-tuning 

```bash
python3 fine-tune-yolo.py
```

## Step 5: Inference 
(Note paths are hardcoded and in progress.)

```bash
python3 inference.py
```