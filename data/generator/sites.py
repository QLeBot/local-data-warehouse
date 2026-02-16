"""
Data generator for sites with coordinates, country information, and equipment IDs.
"""

import random
from typing import List, Optional, Dict
import logging

import pandas as pd
from faker import Faker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker()

# European countries with their ISO country codes
COUNTRY_CODES = {
    'United Kingdom': 'GB',
    'France': 'FR',
    'Germany': 'DE',
    'Spain': 'ES',
    'Italy': 'IT',
    'Netherlands': 'NL',
    'Belgium': 'BE',
    'Portugal': 'PT',
    'Poland': 'PL',
    'Greece': 'GR',
    'Sweden': 'SE',
    'Norway': 'NO',
    'Denmark': 'DK',
    'Finland': 'FI',
    'Switzerland': 'CH',
    'Austria': 'AT',
    'Ireland': 'IE',
    'Czech Republic': 'CZ',
    'Romania': 'RO',
    'Hungary': 'HU'
}

# Country-specific coordinate bounds (latitude, longitude ranges)
# Format: (min_lat, max_lat, min_lon, max_lon)
COUNTRY_BOUNDS = {
    'United Kingdom': (49.8, 60.9, -8.6, 1.8),
    'France': (41.3, 51.1, -5.1, 9.6),
    'Germany': (47.3, 55.1, 5.9, 15.0),
    'Spain': (35.2, 43.8, -9.3, 4.3),
    'Italy': (36.6, 47.1, 6.6, 18.5),
    'Netherlands': (50.7, 53.7, 3.2, 7.2),
    'Belgium': (49.5, 51.5, 2.5, 6.4),
    'Portugal': (36.8, 42.2, -9.5, -6.2),
    'Poland': (49.0, 54.8, 14.1, 24.1),
    'Greece': (34.8, 41.7, 19.4, 28.2),
    'Sweden': (55.3, 69.1, 11.0, 24.2),
    'Norway': (57.9, 80.7, 4.6, 31.3),
    'Denmark': (54.6, 57.8, 8.1, 12.7),
    'Finland': (59.8, 70.1, 20.6, 31.6),
    'Switzerland': (45.8, 47.8, 5.9, 10.5),
    'Austria': (46.4, 49.0, 9.5, 17.2),
    'Ireland': (51.4, 55.4, -10.5, -5.9),
    'Czech Republic': (48.5, 51.1, 12.1, 18.9),
    'Romania': (43.7, 48.2, 20.2, 29.7),
    'Hungary': (45.7, 48.6, 16.1, 22.9)
}


def generate_sites(
    num_sites: int,
    countries: Optional[List[str]] = None,
    num_countries: Optional[int] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a list of sites with coordinates, country information, and equipment IDs.
    
    Args:
        num_sites: Number of sites to generate.
        countries: List of country names to use. If None, randomly selects countries.
        num_countries: Number of countries to randomly select (5-15). Ignored if countries is provided.
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame with columns: site_id (format: countryCode_cityCode_number), country, latitude, longitude, id_solar, id_genset, id_cabinet
    """
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    # Select countries
    if countries is None:
        if num_countries is None:
            num_countries = random.randint(5, 15)
        elif num_countries < 5 or num_countries > 15:
            raise ValueError("num_countries must be between 5 and 15")
        
        available_countries = list(COUNTRY_CODES.keys())
        selected_countries = random.sample(available_countries, min(num_countries, len(available_countries)))
    else:
        selected_countries = countries
        # Validate all countries are in our list
        invalid_countries = [c for c in countries if c not in COUNTRY_CODES]
        if invalid_countries:
            raise ValueError(f"Invalid countries: {invalid_countries}. Available: {list(COUNTRY_CODES.keys())}")
    
    logger.info(f"Generating {num_sites} sites across {len(selected_countries)} countries: {selected_countries}")
    
    # Create reverse mapping: country_code -> country_name for verification
    code_to_country = {code: country for country, code in COUNTRY_CODES.items()}
    
    # Get set of selected country codes for fast lookup
    selected_country_codes = {COUNTRY_CODES[country] for country in selected_countries}
    
    # Track city codes per country to ensure uniqueness in site_id
    city_code_counters: Dict[str, Dict[str, int]] = {}
    
    sites_data = []
    
    # Use default faker for location_on_land (works globally)
    faker_instance = Faker()
    
    for i in range(num_sites):
        # Generate locations on land until we get one in a selected country
        max_attempts = 1000  # Limit attempts to avoid infinite loops
        latitude = None
        longitude = None
        city_name = None
        country = None
        country_code = None
        
        for attempt in range(max_attempts):
            try:
                # Get a location on land
                # location_on_land() returns: (latitude, longitude, place_name, country_code, timezone)
                location = faker_instance.location_on_land()
                
                if location and len(location) >= 4:
                    loc_country_code = location[3]  # Country code is at index 3
                    
                    # Check if this location is in one of our selected countries
                    if loc_country_code in selected_country_codes:
                        # Found a location in a selected country!
                        latitude = float(location[0])
                        longitude = float(location[1])
                        city_name = location[2] if len(location) > 2 and location[2] else None
                        country_code = loc_country_code
                        country = code_to_country.get(loc_country_code, country)
                        break
            except (ValueError, TypeError, IndexError, AttributeError) as e:
                # Continue to next attempt if there's an error
                continue
        
        # If we couldn't find a location in selected countries after max attempts
        if latitude is None or longitude is None or country is None:
            logger.warning(f"Could not find location in selected countries after {max_attempts} attempts for site {i+1}")
            # Fallback: randomly select a country and use a generic location
            country = random.choice(selected_countries)
            country_code = COUNTRY_CODES[country]
            # Use a rough European coordinate range as last resort
            latitude = random.uniform(35.0, 70.0)
            longitude = random.uniform(-10.0, 30.0)
        
        # Get city name if not already obtained
        if city_name is None:
            try:
                # Try to get country-specific city name
                locale_map = {
                    'GB': 'en_GB', 'FR': 'fr_FR', 'DE': 'de_DE', 'ES': 'es_ES',
                    'IT': 'it_IT', 'NL': 'nl_NL', 'BE': 'nl_BE', 'PT': 'pt_PT',
                    'PL': 'pl_PL', 'GR': 'el_GR', 'SE': 'sv_SE', 'NO': 'no_NO',
                    'DK': 'da_DK', 'FI': 'fi_FI', 'CH': 'de_CH', 'AT': 'de_AT',
                    'IE': 'en_IE', 'CZ': 'cs_CZ', 'RO': 'ro_RO', 'HU': 'hu_HU'
                }
                locale = locale_map.get(country_code, 'en_US')
                try:
                    country_fake = Faker(locale=locale)
                    city_name = country_fake.city()
                except (AttributeError, ValueError):
                    city_name = fake.city()
            except Exception:
                city_name = fake.city()
        
        # Final validation - ensure coordinates are valid
        latitude = max(-90, min(90, latitude))
        longitude = max(-180, min(180, longitude))
        
        # Generate city code (first 3 letters of city, uppercase, or abbreviation)
        city_code = city_name[:3].upper().replace(' ', '')
        if len(city_code) < 3:
            city_code = city_code.ljust(3, 'X')
        
        # Initialize country tracking if needed
        if country_code not in city_code_counters:
            city_code_counters[country_code] = {}
        
        # Track site number per country+city combination
        city_key = f"{country_code}_{city_code}"
        if city_key not in city_code_counters[country_code]:
            city_code_counters[country_code][city_key] = 0
        
        city_code_counters[country_code][city_key] += 1
        site_number = city_code_counters[country_code][city_key]
        
        # Generate site_id: countrycode_citycode_number (with underscores)
        site_id = f"{country_code}_{city_code}_{site_number:04d}"
        
        # Generate equipment IDs (using UUIDs for uniqueness)
        id_solar = fake.uuid4()
        id_genset = fake.uuid4()
        id_cabinet = fake.uuid4()
        
        sites_data.append({
            'site_id': site_id,
            'country': country,
            'latitude': latitude,
            'longitude': longitude,
            'id_solar': id_solar,
            'id_genset': id_genset,
            'id_cabinet': id_cabinet
        })
    
    df = pd.DataFrame(sites_data)
    logger.info(f"Generated {len(df)} sites successfully")
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Default: generate 100 sites
    num_sites = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    # Generate sites
    df = generate_sites(
        num_sites=num_sites,
        num_countries=10,  # Randomly select 10 countries
        seed=42  # For reproducibility
    )
    
    # Display sample
    print("\nSample of generated sites:")
    print(df.head(10))
    print(f"\nTotal sites: {len(df)}")
    print(f"\nCountries distribution:")
    print(df['country'].value_counts())
    print(f"\nDataFrame info:")
    print(df.info())
    
    # Save to CSV (optional)
    # output_path = Path(__file__).parent.parent / "raw" / "sites.csv"
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # df.to_csv(output_path, index=False)
    # print(f"\nSaved to {output_path}")
