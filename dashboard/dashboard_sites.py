"""
Streamlit dashboard for visualizing sites on a map.
"""

import sys
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import duckdb

# Add project root to path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from data.generator.sites import generate_sites
except ImportError:
    st.error("Could not import sites generator. Make sure faker is installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Sites Dashboard",
    page_icon="📍",
    layout="wide"
)

st.title("📍 Sites Dashboard")
st.markdown("Visualize sites on an interactive map")

# Initialize session state for storing data
if 'sites_df' not in st.session_state:
    st.session_state.sites_df = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None


@st.cache_data
def load_sites_from_duckdb(
    db_path: Optional[str] = None,
    schema: str = "raw",
    table_name: str = "sites"
) -> Optional[pd.DataFrame]:
    """
    Load sites data from DuckDB warehouse.
    
    Args:
        db_path: Path to DuckDB database file.
        schema: Schema name (raw or bronze).
        table_name: Table name.
    
    Returns:
        DataFrame with sites data or None if not found.
    """
    if db_path is None:
        db_path = str(project_root / "warehouse" / "local.duckdb")
    
    db_file = Path(db_path)
    if not db_file.exists():
        return None
    
    try:
        conn = duckdb.connect(db_path)
        
        # Check if table exists
        query = f"""
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_schema = '{schema}' AND table_name = '{table_name}'
        """
        result = conn.execute(query).fetchone()
        
        if result[0] == 0:
            conn.close()
            return None
        
        # Load data
        full_table_name = f"{schema}.{table_name}"
        df = conn.execute(f"SELECT * FROM {full_table_name}").df()
        conn.close()
        
        return df
    except Exception as e:
        st.warning(f"Error loading from DuckDB: {e}")
        return None


def generate_sites_data(
    num_sites: int,
    num_countries: Optional[int] = None,
    countries: Optional[list] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate sites data.
    
    Args:
        num_sites: Number of sites to generate.
        num_countries: Number of countries to use.
        countries: List of specific countries to use.
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame with generated sites.
    """
    return generate_sites(
        num_sites=num_sites,
        num_countries=num_countries,
        countries=countries,
        seed=seed
    )


def prepare_map_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for st.map by ensuring correct column names and valid coordinates.
    Streamlit's st.map expects 'lat' and 'lon' columns with valid numeric values.
    
    Args:
        df: DataFrame with latitude and longitude columns.
    
    Returns:
        DataFrame with 'lat' and 'lon' columns, filtered to valid coordinates.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    map_df = df.copy()
    
    # Rename columns if needed
    if 'latitude' in map_df.columns and 'lat' not in map_df.columns:
        map_df['lat'] = map_df['latitude']
    if 'longitude' in map_df.columns and 'lon' not in map_df.columns:
        map_df['lon'] = map_df['longitude']
    
    # Ensure lat and lon exist
    if 'lat' not in map_df.columns or 'lon' not in map_df.columns:
        st.error("DataFrame must contain 'latitude'/'lat' and 'longitude'/'lon' columns")
        return pd.DataFrame()
    
    # Convert to numeric, coercing errors to NaN
    map_df['lat'] = pd.to_numeric(map_df['lat'], errors='coerce')
    map_df['lon'] = pd.to_numeric(map_df['lon'], errors='coerce')
    
    # Filter out invalid coordinates
    # Valid latitude: -90 to 90
    # Valid longitude: -180 to 180
    initial_count = len(map_df)
    map_df = map_df.dropna(subset=['lat', 'lon'])
    map_df = map_df[
        (map_df['lat'] >= -90) & (map_df['lat'] <= 90) &
        (map_df['lon'] >= -180) & (map_df['lon'] <= 180)
    ]
    
    if len(map_df) < initial_count:
        invalid_count = initial_count - len(map_df)
        st.warning(f"Filtered out {invalid_count} sites with invalid coordinates")
    
    if len(map_df) == 0:
        st.error("No valid coordinates found in the data")
        return pd.DataFrame()
    
    return map_df


# Sidebar for controls
st.sidebar.header("⚙️ Controls")

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Generate New", "Load from DuckDB"],
    help="Choose to generate new sites or load from DuckDB warehouse",
    key="data_source_radio"
)

# Update session state when data source changes
if st.session_state.data_source != data_source:
    st.session_state.data_source = data_source
    st.session_state.sites_df = None  # Clear data when switching sources

df = None

if data_source == "Load from DuckDB":
    st.sidebar.subheader("DuckDB Settings")
    schema = st.sidebar.selectbox("Schema", ["raw", "bronze"], index=0)
    table_name = st.sidebar.text_input("Table Name", value="sites")
    
    if st.sidebar.button("Load from DuckDB"):
        with st.spinner("Loading sites from DuckDB..."):
            df = load_sites_from_duckdb(schema=schema, table_name=table_name)
            if df is not None:
                st.session_state.sites_df = df
                st.sidebar.success(f"Loaded {len(df)} sites from DuckDB")
            else:
                st.sidebar.error("No sites found in DuckDB. Try generating new sites.")
                st.session_state.sites_df = None
    
    # Use session state data if available, otherwise try to load
    if st.session_state.sites_df is not None:
        df = st.session_state.sites_df
    elif df is None:
        df = load_sites_from_duckdb(schema=schema, table_name=table_name)
        if df is not None:
            st.session_state.sites_df = df

else:  # Generate New
    st.sidebar.subheader("Generation Settings")
    
    num_sites = st.sidebar.slider(
        "Number of Sites",
        min_value=10,
        max_value=100000,
        value=100,
        step=100
    )
    
    use_specific_countries = st.sidebar.checkbox("Select Specific Countries", value=False)
    
    if use_specific_countries:
        available_countries = [
            'United Kingdom', 'France', 'Germany', 'Spain', 'Italy',
            'Netherlands', 'Belgium', 'Portugal', 'Poland', 'Greece',
            'Sweden', 'Norway', 'Denmark', 'Finland', 'Switzerland',
            'Austria', 'Ireland', 'Czech Republic', 'Romania', 'Hungary'
        ]
        selected_countries = st.sidebar.multiselect(
            "Countries",
            available_countries,
            default=available_countries[:5]
        )
        countries = selected_countries if selected_countries else None
        num_countries = None
    else:
        num_countries = st.sidebar.slider(
            "Number of Countries",
            min_value=5,
            max_value=15,
            value=10,
            step=1
        )
        countries = None
    
    seed = st.sidebar.number_input(
        "Random Seed (for reproducibility)",
        min_value=0,
        value=42,
        step=1
    )
    
    if st.sidebar.button("Generate Sites", type="primary"):
        with st.spinner(f"Generating {num_sites} sites..."):
            df = generate_sites_data(
                num_sites=num_sites,
                num_countries=num_countries,
                countries=countries,
                seed=seed
            )
            # Store in session state to ensure map refreshes
            st.session_state.sites_df = df
            st.sidebar.success(f"Generated {len(df)} sites!")
            # Force rerun to update the map
            st.rerun()
    
    # Use session state data if available, otherwise generate default
    if st.session_state.sites_df is not None:
        df = st.session_state.sites_df
    elif df is None:
        with st.spinner("Generating default sites..."):
            df = generate_sites_data(
                num_sites=100,
                num_countries=10,
                seed=42
            )
            st.session_state.sites_df = df

# Main content
if df is not None and len(df) > 0:
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sites", len(df))
    
    with col2:
        st.metric("Countries", df['country'].nunique())
    
    with col3:
        st.metric("Unique Site IDs", df['site_id'].nunique())
    
    with col4:
        st.metric("Equipment IDs", len(df) * 3)  # solar, genset, cabinet
    
    st.divider()
    
    # Filters
    st.subheader("🔍 Filters")
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        selected_countries = st.multiselect(
            "Filter by Country",
            options=sorted(df['country'].unique()),
            default=sorted(df['country'].unique())
        )
    
    with filter_col2:
        search_site_id = st.text_input("Search Site ID", placeholder="e.g., USNYC")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    
    if search_site_id:
        filtered_df = filtered_df[
            filtered_df['site_id'].str.contains(search_site_id, case=False, na=False)
        ]
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} sites")
    
    # Debug: Show coordinate statistics
    with st.expander("🔍 Coordinate Debug Info"):
        if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
            coord_df = filtered_df[['site_id', 'country', 'latitude', 'longitude']].copy()
            coord_df['latitude'] = pd.to_numeric(coord_df['latitude'], errors='coerce')
            coord_df['longitude'] = pd.to_numeric(coord_df['longitude'], errors='coerce')
            
            st.write("**Coordinate Statistics:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Latitude range: {coord_df['latitude'].min():.4f} to {coord_df['latitude'].max():.4f}")
                st.write(f"Longitude range: {coord_df['longitude'].min():.4f} to {coord_df['longitude'].max():.4f}")
            with col2:
                st.write(f"Invalid latitudes: {coord_df['latitude'].isna().sum()}")
                st.write(f"Invalid longitudes: {coord_df['longitude'].isna().sum()}")
            
            # Show sample coordinates
            st.write("**Sample Coordinates:**")
            st.dataframe(coord_df.head(10), width='stretch', hide_index=True)
    
    # Map visualization
    st.subheader("🗺️ Sites Map")
    
    if len(filtered_df) > 0:
        map_df = prepare_map_data(filtered_df)
        
        if len(map_df) > 0:
            # Select columns to display on map
            display_cols = ['site_id', 'country', 'lat', 'lon']
            if 'id_solar' in map_df.columns:
                display_cols.append('id_solar')
            
            # Ensure we have the required columns
            available_cols = [col for col in display_cols if col in map_df.columns]
            if 'lat' in map_df.columns and 'lon' in map_df.columns:
                st.map(
                    map_df[available_cols],
                    latitude='lat',
                    longitude='lon',
                    size=100,
                    color='#FF0000',
                    zoom=2
                )
            else:
                st.error(f"Missing required columns for map. Found: {list(map_df.columns)}")
        else:
            st.error("No valid coordinate data to display on map")
    else:
        st.warning("No sites match the selected filters")
    
    st.divider()
    
    # Data table
    st.subheader("📊 Sites Data")
    
    # Show/hide columns
    default_cols = ['site_id', 'country', 'latitude', 'longitude']
    all_cols = list(df.columns)
    available_cols = [col for col in all_cols if col not in default_cols]
    
    if available_cols:
        selected_additional_cols = st.multiselect(
            "Additional Columns to Display",
            available_cols,
            default=available_cols[:3] if len(available_cols) >= 3 else available_cols
        )
        display_columns = default_cols + selected_additional_cols
    else:
        display_columns = default_cols
    
    st.dataframe(
        filtered_df[display_columns],
        width='stretch',
        hide_index=True
    )
    
    # Country distribution chart
    st.subheader("📈 Country Distribution")
    country_counts = filtered_df['country'].value_counts().sort_values(ascending=True)
    st.bar_chart(country_counts)
    
    # Download button
    st.divider()
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="sites_data.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Use the sidebar controls to generate or load sites data")
    
    # Show instructions
    with st.expander("ℹ️ How to use this dashboard"):
        st.markdown("""
        ### Generate New Sites
        1. Select "Generate New" in the sidebar
        2. Adjust the number of sites and countries
        3. Click "Generate Sites" button
        
        ### Load from DuckDB
        1. Select "Load from DuckDB" in the sidebar
        2. Specify the schema and table name
        3. Click "Load from DuckDB" button
        
        ### Features
        - **Interactive Map**: View all sites on a world map
        - **Filters**: Filter by country or search by site ID
        - **Statistics**: View summary metrics
        - **Data Table**: Browse the full dataset
        - **Download**: Export filtered data as CSV
        """)
