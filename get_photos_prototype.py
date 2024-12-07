from google.oauth2 import service_account
import google.auth.transport.requests
import requests
import os
import json

def get_access_token():
    try:
        # Print the contents of the JSON file (excluding sensitive data)
        with open('pocketfinder-a0ced62ce802.json', 'r') as f:
            creds_data = json.load(f)
            #print(f"Project ID: {creds_data.get('project_id')}")
            #print(f"Client Email: {creds_data.get('client_email')}")
        
        credentials = service_account.Credentials.from_service_account_file(
            'pocketfinder-a0ced62ce802.json',
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

def search_place():
    try:
        access_token = get_access_token()
        
        base_url = "https://places.googleapis.com/v1/places:searchText"
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'X-Goog-FieldMask': 'places.id,places.displayName,places.formattedAddress'
        }
        
        data = {
            "textQuery": "Archer Hotel Burlington MA"
        }
        
        response = requests.post(
            base_url,
            headers=headers,
            json=data
        )
        
        print(f"\nResponse status: {response.status_code}")
        print(f"Response body: {response.text}")
        
    except Exception as e:
        print(f"Exception in search_place: {str(e)}")

def get_hotel_photos():
    # Get access token from service account
    access_token = get_access_token()

    with open('api_key.txt', 'r') as file:
        api_key = file.read().strip()
    
    # API endpoint for Places API (New)
    base_url = "https://places.googleapis.com/v1/"
    place_id = "ChIJ219n706e44kRnQU0tC1nqo0"  # Archer Hotel Burlington
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
        for i, photo in enumerate(place_data['photos']):
            photo_name = photo['name'] 
            photo_url = f"{base_url}{photo_name}/media?key={api_key}&maxHeightPx=4032&maxWidthPx=4032"
            print(f"Requesting media from: {photo_url}")

            # photo_url = f"{base_url}{place_id}/photos/{photo['name']}/media"
            photo_response = requests.get(
                photo_url,
                headers=headers2,
                allow_redirects=True  # Explicitly allow redirects
            )
            #import pdb; pdb.set_trace()
            if photo_response.status_code == 200:
                # Save the photo
                filename = f"photo_{i}.jpg"
                file_path = os.path.join('hotel_photos', filename)
                
                with open(file_path, 'wb') as f:
                    f.write(photo_response.content)
                print(f"Saved photo {i} to {file_path}")
                photo_results.append(file_path)
            else:
                print(f"Error downloading photo {i}: {response.text}")
        
        print(photo_results)

        return photo_results
       
if __name__ == "__main__":
    # search_place()
    get_hotel_photos()
