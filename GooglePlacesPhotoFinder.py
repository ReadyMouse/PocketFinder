from google.oauth2 import service_account
import google.auth.transport.requests
import requests
import os
import json
import time

def get_access_token(self):
    try:
        # Print the contents of the JSON file (excluding sensitive data)
        #with open(self.cred_json_path, 'r') as f:
        #    creds_data = json.load(f)
            #print(f"Project ID: {creds_data.get('project_id')}")
            #print(f"Client Email: {creds_data.get('client_email')}")
        
        credentials = service_account.Credentials.from_service_account_file(
            self.cred_json_path,
            scopes=['https://www.googleapis.com/auth/maps-platform.places']
        )
        
        # Print credential info
        #print(f"\nCredentials info:")
        #print(f"Valid: {credentials.valid}")
        #print(f"Expired: {credentials.expired}")
        #print(f"Scopes: {credentials.scopes}")
        
        # Force token refresh
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        
        token = credentials.token
        print(f"\nToken obtained: {token[:10]}...")  # Show first 10 chars
        return token
        
    except Exception as e:
        print(f"Error getting token: {str(e)}")
        raise

class GooglePlacesPhotoFinder():
    """Implementation using Google Places API"""
    def __init__(self, 
                 cred_json_path ='pocketfinder-a0ced62ce802.json', 
                 api_key='api_key.txt'):
        
        self.cred_json_path = cred_json_path

        with open(api_key, 'r') as file:
            self.api_key = file.read().strip()

    def get_place_photos(self, place_id='ChIJ219n706e44kRnQU0tC1nqo0'):
        # Get access token from service account
        access_token = get_access_token(self)

        # API endpoint for Places API (New)
        base_url = "https://places.googleapis.com/v1/"
        # place_id = "ChIJ219n706e44kRnQU0tC1nqo0"  # Archer Hotel Burlington
        place_url = f"{base_url}places/{place_id}"
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            #'X-Goog-Api-Key': api_key,  # Leave empty when using OAuth
            'X-Goog-FieldMask': 'photos'
        }

        headers2 = {
            'Authorization': f'Bearer {access_token}',
            #'X-Goog-Api-Key': api_key,  # Leave empty when using OAuth
            #'X-Goog-FieldMask': 'photos'
        }
        
        # Get place details with photos
        response = requests.get(place_url, headers=headers)
        
        if response.status_code == 200:
            place_data = response.json()
            print("\nPlace data received:")
            #print(json.dumps(place_data, indent=2))
            
            # Create photos directory if it doesn't exist
            os.makedirs('hotel_photos', exist_ok=True)

            if 'photos' not in place_data:
                return "No photos available for this place"
                return []
        
            # Get the actual photo media for each photo reference
            # 
            photo_results = []
            page_count = 0 
            next_page_token = None

            while True:
                # Add page token to URL if we have one
                url = place_url
                if next_page_token:
                    url = f"{url}?pageToken={next_page_token}"
                
                print(f"\nFetching page {page_count + 1}...")
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    print(f"Error getting place details: {response.text}")
                    break
                    
                place_data = response.json()
                print(f"Received {len(place_data.get('photos', []))} photos in this page")
                
                if 'photos' not in place_data:
                    print("No photos in this page")
                    break

                for i, photo in enumerate(place_data['photos']):
                    photo_name = photo['name'] 
                    photo_url = f"{base_url}{photo_name}/media?key={self.api_key}&maxHeightPx=4032&maxWidthPx=4032"
                    # print(f"Requesting media from: {photo_url}")

                    # photo_url = f"{base_url}{place_id}/photos/{photo['name']}/media"
                    photo_response = requests.get(
                        photo_url,
                        headers=headers2,
                        allow_redirects=True,  # Explicitly allow redirects
                        stream=True
                    )
                    #import pdb; pdb.set_trace()
                    if photo_response.status_code == 200:
                        # Save the photo
                        total_index = len(photo_results)
                        filename = f"photo_{total_index}.jpg"
                        file_path = os.path.join('hotel_photos', filename)
                        
                        with open(file_path, 'wb') as f:
                            f.write(photo_response.content)

                        print(f"Saved photo {i} to {file_path}")
                        photo_results.append(file_path)
                    else:
                        print(f"Error downloading photo {i}: {response.text}")
                
                # Check for next page
                next_page_token = place_data.get('nextPageToken')
                if not next_page_token:
                    print("\nNo more pages available")
                    break
                
                page_count += 1
                print(f"Found next page token: {next_page_token[:20]}...")
                
                # Optional: Add a small delay between pages to avoid rate limiting
                time.sleep(1)

            print(f"\nDownload complete! Successfully downloaded {len(photo_results)} photos across {page_count + 1} pages")
            return photo_results
        
if __name__ == "__main__":
    # search_place()
    finder = GooglePlacesPhotoFinder()
    finder.get_place_photos()
