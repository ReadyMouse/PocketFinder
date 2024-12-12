from google.oauth2 import service_account
import google.auth.transport.requests
import requests
import os
import json
import time

def get_access_token(self):
    try:
        credentials = service_account.Credentials.from_service_account_file(
            self.cred_json_path,
            scopes=['https://www.googleapis.com/auth/maps-platform.places']
        )
        
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        
        token = credentials.token
        # print(f"\nToken obtained: {token[:10]}...")
        return token
        
    except Exception as e:
        print(f"Error getting token: {str(e)}")
        raise

class GooglePlacesPhotoReviews:
    def __init__(self, 
                 cred_json_path='pocketfinder-a0ced62ce802.json', 
                 api_key='api_key.txt',
                 output_dir = 'google_reviews'):
        
        self.cred_json_path = cred_json_path
        with open(api_key, 'r') as file:
            self.api_key = file.read().strip()
        self.base_url = "https://places.googleapis.com/v1/"

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_place_details(self, place_id='ChIJ219n706e44kRnQU0tC1nqo0'):
        """Get place details with optional pagination"""
        access_token = get_access_token(self)
        place_url = f"{self.base_url}places/{place_id}"
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'X-Goog-FieldMask': 'photos,reviews'
        }

        response = requests.get(place_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error getting place details: {response.text}")
            return None
            
        return response.json()

    def download_photo(self, photo_name, save_name, prefix=""):
        """Helper method to download a single photo"""
        headers = {'Authorization': f'Bearer {get_access_token(self)}'}
        photo_url = f"{self.base_url}{photo_name}/media?key={self.api_key}&maxHeightPx=4032&maxWidthPx=4032"

        photo_response = requests.get(
            photo_url,
            headers=headers,
            allow_redirects=True,
            stream=True
        )
        file_path = os.path.join(self.output_dir, save_name)

        if photo_response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(photo_response.content)
            # print(f"Saved {save_name} to {self.output_dir}")
            return file_path
        else:
            print(f"Error downloading photo: {photo_response.text}")
            return None

    def get_place_photos(self, place_id='ChIJ219n706e44kRnQU0tC1nqo0'):
        """Get and download photos for a place"""
        place_data = self.get_place_details(place_id)
        if not place_data or 'photos' not in place_data:
            return "No photos available for this place"
        
        #os.makedirs('google_photos', exist_ok=True)
        photo_results = []
        
        for i, photo in enumerate(place_data['photos']):
            photo_name = photo['name']
            save_name = f"photo_{place_id}_{i}.jpg"
            
            downloaded_path = self.download_photo(photo_name, save_name, "place")
            if downloaded_path:
                photo_results.append(downloaded_path)
                
        print(f"\nDownload complete! Successfully downloaded {len(photo_results)} place photos")
        return photo_results

    def get_place_reviews(self, place_id='ChIJ219n706e44kRnQU0tC1nqo0'):
        """Get all reviews and their photos for a place with pagination"""
        place_data = self.get_place_details(place_id)
        if not place_data:
            return "Failed to get place details"
        
        reviews = place_data.get('reviews', [])
        all_reviews = []

            
        for i, review in enumerate(reviews):

            processed_review = {
                'text': review['text']['text'],
                'publish_time': review.get('publishTime'),
                'photos': []
            }
            
            # Handle review photos if they exist
            if 'photos' in review:
                for j, photo in enumerate(review['photos']):
                    photo_name = photo['name']
                    file_path = os.path.join(self.output_dir, 
                                            f"review_{place_id}_photo_{j}.jpg")
                    
                    downloaded_path = self.download_photo(photo_name, file_path, "review")
                    if downloaded_path:
                        processed_review['photos'].append({
                            'local_path': downloaded_path,
                            'photo_name': photo_name
                        })
            
            all_reviews.append(processed_review)
            
        reviews_data = {
            'reviews': all_reviews
        }
        
        # Save reviews data to JSON
        output_file = os.path.join(self.output_dir, f'reviews_{place_id}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reviews_data, f, ensure_ascii=False, indent=2)

        return reviews_data

if __name__ == "__main__":
    client = GooglePlacesPhotoReviews()
    # Get all reviews (or specify max_reviews parameter to limit)
    photos = client.get_place_photos()
    reviews = client.get_place_reviews()  
