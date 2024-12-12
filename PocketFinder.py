import requests
from typing import List, Dict
import time
from dataclasses import dataclass
from GooglePlacesPhotoReviews import GooglePlacesPhotoReviews
from PoolTableInference import PoolTableInference
from GooglePlacesVenueFinder import GooglePlacesVenueFinder
import os 
import pdb

# Note this is unfinished, but a holding spot as I finished modules 
# M. Bennett December 2024

def main():
    # Example usage
    cred_json_path ='pocketfinder-a0ced62ce802.json' 
    api_key='api_key.txt'
    output_dir = '/PocketFinder/nearby_NYC'
    os.makedirs(output_dir, exist_ok=True)

    # Example coordinates for New York City
    # TODO: Make place type an input field. Resturant, bar, etc
    venue_Finder = GooglePlacesVenueFinder()
    venues = venue_Finder.getPlaceID(latitude=40.7128,
                           longitude=-74.0060,
                           radius=500)


    # TODO: How many venues did it find? 
    # print(f'Found {len(venues.items)} venues.')

    # Initialize photo finder
    photo_finder = GooglePlacesPhotoReviews(
        cred_json_path=cred_json_path,
        api_key=api_key,
        output_dir=output_dir
    )

    # Intialuze the inference engine
    pool_table_inference = PoolTableInference(
        model_path = './First-working-copy/weights/best.pt',
        conf_threshold=0.5,
        output_dir=output_dir
    )

    # Process each venue category
    for category, venue_list in venues.items():
        for venue in venue_list:
            print('-----------------')
            print(f"Processing venue: {venue.name}")
            place_id = venue.place_id  # Fixed attribute name
            # Get photos and reviews
            try:
                photos = photo_finder.get_place_photos(place_id)
                reviews = photo_finder.get_place_reviews(place_id)
                
                # Make sure venue object has these attributes
                if not hasattr(venue, 'photos_urls'):
                    venue.photos_urls = []
                if not hasattr(venue, 'reviews'):
                    venue.reviews = []
                    
                venue.photos_urls.extend(photos)
                venue.reviews.extend(reviews)
                
                print(f"Number of photos found: {len(photos)}")
                print(f"Number of reviews found: {len(reviews)}")
                
                # Rate limiting
                time.sleep(0.2)

                # Process images for pool tables if photos exist
                table_prob = []
                if venue.photos_urls:
                    try:
                        for item in venue.photos_urls:
                            results = pool_table_inference.run_inference(image_path=item,
                                                                        save_negative=False # to save photos that don't have pool tables 
                                                                        )   
                            #pdb.set_trace()
                            if results[os.path.basename(item)]['class_name'] == 'pool_table':
                                confidence = results[os.path.basename(item)]['confidence']
                            else:
                                confidence = 1 - results[os.path.basename(item)]['confidence']
                                
                            table_prob.append(confidence)

                        # TODO: math on the set of probabilities of a pool table 
                        max_conf = max(table_prob)
                        print(f'There is a {max_conf:.1%} chance that {venue.name} has a pool table.')

                    except Exception as e:
                        print(f"Error in pool table inference for {venue.name}: {str(e)}")

                # TODO: Add text review processing
                
            except Exception as e:
                print(f"Error processing venue {venue.name}: {str(e)}")
            continue
    
if __name__ == "__main__":
    main()
