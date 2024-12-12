import requests
from typing import List, Dict
import time
from dataclasses import dataclass
import GooglePlacesPhotoReviews
import PoolTableInference

# Note this is unfinished, but a holding spot as I finished modules 
# M. Bennett December 2024

def main():
    # Example usage
    cred_json_path ='pocketfinder-a0ced62ce802.json', 
    api_key='api_key.txt'

    # TODO: venue_finder = GooglePlacesVenueFinder()
    # TODO: paths are hardcoded to make new folder
    photo_finder = GooglePlacesPhotoReviews(cred_json_path, api_key)
    
    
    # Example coordinates (replace with actual location)
    latitude = 40.7128
    longitude = -74.0060
    radius = 1000  # meters
    
    # TODO find the place IDs 
    # venues = venue_finder.find_venues(latitude, longitude, radius)
    
    for venue in venues:
        print(f"Found venue: {venue.name}") 
        place_ID = venue.ID
        photos = photo_finder.get_place_photos(place_ID)
        reviews = photo_finder.get_place_reviews(place_ID) # saves photos in same folder
        venue.photos_urls.extend(photos)
        venue.reviews.extend(reviews)
        print(f"Number of photos found: {len(photos)}")
        print(f"Number of reviews found: {len(reviews)}")
        time.sleep(0.2)  # Additional rate limiting
    
    # Image Processing 
    pool_table_inference = PoolTableInference(image_path= venue.photos_urls,
                                              output_dir = '', 
                                              conf_threshold=0.5)
    results = pool_table_inference.run_inference
    print(results)

if __name__ == "__main__":
    main()
