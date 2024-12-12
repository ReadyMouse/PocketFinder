from google.oauth2 import service_account
import google.auth.transport.requests
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass

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
    
@dataclass
class Venue:
    place_id: str
    name: str
    types: List[str]
    rating: Optional[float]
    vicinity: str

class GooglePlacesVenueFinder: 
    def __init__(self, 
                cred_json_path='pocketfinder-a0ced62ce802.json', 
                api_key='api_key.txt'
                ):
    
        self.cred_json_path = cred_json_path
        with open(api_key, 'r') as file:
            self.api_key = file.read().strip()
        self.base_url = "https://places.googleapis.com/v1/"

    def getPlaceID(self, latitude=42.36, longitude=71.05, radius=10000):
        """
        Finds venues near a given latitude/longitude using Google Places API.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            radius (int): Search radius in meters (max 50000)
        """
        # Input validation
        if not (-90 <= latitude<= 90) or not (-180 <= longitude <= 180):
            raise ValueError("Invalid coordinates")
        if not (0 < radius <= 50000):
            raise ValueError("Radius must be between 1 and 50000 meters")
            
        # Define search types and initialize results
        venue_types = {
            'hotels': ['lodging'],
            'restaurants': ['restaurant'],
            'bars': ['bar', 'night_club'],
            'sports': ['bowling_alley']
        }
        
        results = {category: [] for category in venue_types}
        base_url = "https://places.googleapis.com/v1/places:searchNearby"
        
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.displayName,places.id,places.types,places.rating,places.formattedAddress"
        }
        
        # Search for each venue type
        for category, types in venue_types.items():
            for venue_type in types:
                payload = {
                    "locationRestriction": {
                        "circle": {
                            "center": {
                                "latitude": latitude,
                                "longitude": longitude
                            },
                            "radius": float(radius)
                        }
                    },
                    "includedTypes": [venue_type]
                }
                
                try:
                    response = requests.post(base_url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'places' in data:
                        for place in data['places']:
                            venue = Venue(
                                place_id=place.get('id', ''),
                                name=place.get('displayName', {}).get('text', ''),
                                types=place.get('types', []),
                                rating=place.get('rating'),
                                vicinity=place.get('formattedAddress', '')
                            )
                            if venue not in results[category]:
                                results[category].append(venue)
                                
                except requests.RequestException as e:
                    print(f"Error fetching {venue_type} venues: {str(e)}")
                    continue
                    
        return results

# Example usage:
if __name__ == "__main__":
    # Example coordinates for New York City
    venueFinder = GooglePlacesVenueFinder()
    venues = venueFinder.getPlaceID(latitude=40.7128,
                           longitude=-74.0060,
                           radius=1000)

    # Print results
    for category, venue_list in venues.items():
        print(f"\n{category.upper()}:")
        for venue in venue_list:
            print(f"- {venue.name} (ID: {venue.place_id})")
            print(f"  Address: {venue.vicinity}")
            if venue.rating:
                print(f"  Rating: {venue.rating}")
