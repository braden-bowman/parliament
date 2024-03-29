"""

https://www.geonames.org/export/web-services.html

"""
import requests
from getpass import getpass, getuser



# xml return
# url = f"http://api.geonames.org/findNearbyPlaceName?lat=47.3&lng=9&username={account}"

def get_PlaceName(
    latitude,
    longitude,
    verbose=False
):
    """
    requires a user name and passwork from http://api.geonames.org
    you must also enable teh query service from your account on their site
    """
    password = getpass()
    account = getuser()
    
    # url = f"http://api.geonames.org/findNearbyPlaceNameJSON?lat={latitude}&lng={longitude}&username={account}"
    url = f"""http://api.geonames.org/findNearbyPlaceNameJSON?lat={latitude}&lng={longitude}&radius=300&username={account}"""
    response = requests.get(url)

    if verbose==True:
        print(type(latitude), latitude)
        print(url)
        
    if response.status_code == 200:
        return response.json()
    
    else:
        return response.text
