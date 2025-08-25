import streamlit as st
import cv2
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS as TIFF_TAGS
import io
import base64

class GeoTIFFGeolocation:
    def __init__(self, image_path=None, image_data=None, corner_gps_coords=None):
        """
        Initialize with a TIFF file and optional corner GPS coordinates
        
        Args:
            image_path: Path to the TIFF file
            image_data: PIL Image object (for uploaded files)
            corner_gps_coords: List of 4 GPS coordinates [(lon1,lat1), (lon2,lat2), (lon3,lat3), (lon4,lat4)]
                              Order: top-left, top-right, bottom-right, bottom-left
        """
        self.image_path = image_path
        self.image = None
        self.pil_image = None
        self.homography_matrix = None
        
        # Load the image
        if image_data:
            self._load_image_from_data(image_data)
        elif image_path:
            self._load_image()
        
        if corner_gps_coords:
            self._setup_homography_from_corners(corner_gps_coords)
        else:
            self._setup_homography_from_geotiff()
    
    def _load_image_from_data(self, image_data):
        """Load the TIFF image from uploaded data"""
        try:
            self.pil_image = image_data
            # Convert PIL to numpy array for OpenCV compatibility
            self.image = np.array(self.pil_image)
            
            # Handle different image formats
            if len(self.image.shape) == 3 and self.image.shape[2] == 3:
                # RGB image - convert to BGR for OpenCV
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            elif len(self.image.shape) == 3 and self.image.shape[2] == 4:
                # RGBA image - convert to BGR
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGR)
            
            self.height, self.width = self.image.shape[:2]
            
        except Exception as e:
            raise ValueError(f"Could not load image from data: {e}")
    
    def _load_image(self):
        """Load the TIFF image using PIL"""
        try:
            self.pil_image = Image.open(self.image_path)
            # Also load with OpenCV for compatibility with existing code
            self.image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if self.image is None:
                # If OpenCV fails, convert PIL to numpy array
                self.image = np.array(self.pil_image)
            
            self.height, self.width = self.image.shape[:2]
            
        except Exception as e:
            raise ValueError(f"Could not load image from {self.image_path}: {e}")
    
    def _setup_homography_from_corners(self, corner_gps_coords):
        """Setup homography using provided corner coordinates"""
        if len(corner_gps_coords) != 4:
            raise ValueError("Exactly 4 corner coordinates required")
        
        # Image corner points (top-left, top-right, bottom-right, bottom-left)
        img_points = np.float32([
            [0, 0],                           # top-left
            [self.width-1, 0],                # top-right
            [self.width-1, self.height-1],    # bottom-right
            [0, self.height-1]                # bottom-left
        ])
        
        # Convert GPS coordinates to numpy array
        gps_points = np.float32(corner_gps_coords)
        
        # Calculate homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(img_points, gps_points)
        return True
    
    def _setup_homography_from_geotiff(self):
        """Setup homography using existing GeoTIFF georeference data from PIL"""
        try:
            # Read TIFF metadata using PIL
            metadata = self._extract_geotiff_metadata()
            
            if not metadata:
                return False
            
            # Extract georeferencing information
            if 'corner_coords' in metadata:
                # Use direct corner coordinates if available
                corner_coords = metadata['corner_coords']
                self._setup_homography_from_corners(corner_coords)
                return True
                
            elif 'geo_transform' in metadata and 'width' in metadata and 'height' in metadata:
                # Calculate corner coordinates from transform matrix
                transform = metadata['geo_transform']
                width = metadata['width']
                height = metadata['height']
                
                # Calculate the four corners
                corners = []
                for px, py in [(0, 0), (width, 0), (width, height), (0, height)]:
                    x = transform[0] + px * transform[1] + py * transform[2]
                    y = transform[3] + px * transform[4] + py * transform[5]
                    corners.append([x, y])
                
                self._setup_homography_from_corners(corners)
                return True
            
            else:
                return False
                
        except Exception as e:
            st.error(f"Could not read GeoTIFF georeference data: {e}")
            return False
    
    def _extract_geotiff_metadata(self):
        """Extract georeference metadata from TIFF using PIL"""
        metadata = {}
        
        try:
            # Get basic image info
            metadata['width'] = self.pil_image.width
            metadata['height'] = self.pil_image.height
            
            # Try to get TIFF tags
            if hasattr(self.pil_image, 'tag_v2'):
                tags = self.pil_image.tag_v2
                
                # Look for GeoTIFF tags
                geo_tags = {}
                
                # Common GeoTIFF tag IDs
                geotiff_tags = {
                    33550: 'ModelPixelScaleTag',
                    33922: 'ModelTiepointTag',
                    34264: 'ModelTransformationTag',
                    34735: 'GeoKeyDirectoryTag',
                    34736: 'GeoDoubleParamsTag',
                    34737: 'GeoAsciiParamsTag'
                }
                
                for tag_id, tag_name in geotiff_tags.items():
                    if tag_id in tags:
                        geo_tags[tag_name] = tags[tag_id]
                
                # Parse ModelTiepointTag specifically
                if 'ModelTiepointTag' in geo_tags:
                    tiepoints = geo_tags['ModelTiepointTag']
                    
                    # Parse all tiepoints
                    num_tiepoints = len(tiepoints) // 6
                    
                    if num_tiepoints >= 2:
                        # Extract all tiepoints
                        parsed_tiepoints = []
                        for i in range(num_tiepoints):
                            start_idx = i * 6
                            tp = {
                                'pixel_x': tiepoints[start_idx],
                                'pixel_y': tiepoints[start_idx + 1],
                                'pixel_z': tiepoints[start_idx + 2],
                                'geo_x': tiepoints[start_idx + 3],
                                'geo_y': tiepoints[start_idx + 4],
                                'geo_z': tiepoints[start_idx + 5]
                            }
                            parsed_tiepoints.append(tp)
                        
                        # Find corner tiepoints
                        top_left = None
                        top_right = None
                        bottom_left = None
                        bottom_right = None
                        
                        for tp in parsed_tiepoints:
                            px, py = tp['pixel_x'], tp['pixel_y']
                            if px <= 1 and py <= 1:
                                top_left = tp
                            elif px >= self.width - 1 and py <= 1:
                                top_right = tp
                            elif px <= 1 and py >= self.height - 1:
                                bottom_left = tp
                            elif px >= self.width - 1 and py >= self.height - 1:
                                bottom_right = tp
                        
                        # Use corner coordinates if all found
                        if top_left and top_right and bottom_left and bottom_right:
                            corner_coords = [
                                [top_left['geo_x'], top_left['geo_y']],      # top-left
                                [top_right['geo_x'], top_right['geo_y']],    # top-right
                                [bottom_right['geo_x'], bottom_right['geo_y']], # bottom-right
                                [bottom_left['geo_x'], bottom_left['geo_y']]  # bottom-left
                            ]
                            
                            metadata['corner_coords'] = corner_coords
            
            return metadata
            
        except Exception as e:
            return {}
    
    def pixel_to_gps(self, pixel_x, pixel_y):
        """Convert pixel coordinates to GPS coordinates"""
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not initialized")
        
        point = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
        gps_point = cv2.perspectiveTransform(point, self.homography_matrix)
        
        longitude = float(gps_point[0][0][0])
        latitude = float(gps_point[0][0][1])
        
        return longitude, latitude
    
    def get_display_image(self, max_width=800):
        """Get image for display with proper scaling"""
        if self.image is None:
            return None
        
        # Calculate scaling factor
        scale_factor = min(max_width / self.width, max_width / self.height)
        if scale_factor > 1:
            scale_factor = 1
        
        # Resize image
        new_width = int(self.width * scale_factor)
        new_height = int(self.height * scale_factor)
        
        display_image = cv2.resize(self.image, (new_width, new_height))
        
        # Convert BGR to RGB for display
        if len(display_image.shape) == 3:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        return display_image, scale_factor

def main():
    st.set_page_config(page_title="GeoTIFF Interactive Viewer", layout="wide")
    
    st.title("🌍 Interactive GeoTIFF Viewer")
    st.write("Upload a GeoTIFF file and click on it to see GPS coordinates at that location.")
    
    # Initialize session state
    if 'geo_locator' not in st.session_state:
        st.session_state.geo_locator = None
    if 'display_image' not in st.session_state:
        st.session_state.display_image = None
    if 'scale_factor' not in st.session_state:
        st.session_state.scale_factor = 1
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload GeoTIFF file",
            type=['tif', 'tiff'],
            help="Upload a GeoTIFF file with embedded georeference data or provide corner coordinates manually."
        )
        
        # Manual corner coordinates input
        st.subheader("Manual Corner Coordinates")
        st.write("If your TIFF doesn't have georeference data, provide corner GPS coordinates:")
        
        use_manual = st.checkbox("Use manual coordinates")
        
        corner_coords = None
        if use_manual:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top-Left Corner**")
                tl_lon = st.number_input("Longitude", value=103.8198, format="%.6f", key="tl_lon")
                tl_lat = st.number_input("Latitude", value=1.3521, format="%.6f", key="tl_lat")
                
                st.write("**Bottom-Left Corner**")
                bl_lon = st.number_input("Longitude", value=103.8198, format="%.6f", key="bl_lon")
                bl_lat = st.number_input("Latitude", value=1.3200, format="%.6f", key="bl_lat")
            
            with col2:
                st.write("**Top-Right Corner**")
                tr_lon = st.number_input("Longitude", value=103.8500, format="%.6f", key="tr_lon")
                tr_lat = st.number_input("Latitude", value=1.3521, format="%.6f", key="tr_lat")
                
                st.write("**Bottom-Right Corner**")
                br_lon = st.number_input("Longitude", value=103.8500, format="%.6f", key="br_lon")
                br_lat = st.number_input("Latitude", value=1.3200, format="%.6f", key="br_lat")
            
            corner_coords = [
                (tl_lon, tl_lat),  # top-left
                (tr_lon, tr_lat),  # top-right
                (br_lon, br_lat),  # bottom-right
                (bl_lon, bl_lat)   # bottom-left
            ]
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load the image
            pil_image = Image.open(uploaded_file)
            
            # Create GeoTIFF locator
            geo_locator = GeoTIFFGeolocation(
                image_data=pil_image,
                corner_gps_coords=corner_coords if use_manual else None
            )
            
            # Check if georeference data was found
            has_geo = geo_locator.homography_matrix is not None
            
            if has_geo:
                st.success("✅ Georeference data loaded successfully!")
            else:
                st.error("❌ No georeference data found. Please provide manual corner coordinates.")
                st.stop()
            
            # Store in session state
            st.session_state.geo_locator = geo_locator
            
            # Get display image
            display_image, scale_factor = geo_locator.get_display_image(max_width=800)
            st.session_state.display_image = display_image
            st.session_state.scale_factor = scale_factor
            
            # Display image info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Image Size", f"{geo_locator.width} × {geo_locator.height}")
            with col2:
                st.metric("Display Scale", f"{scale_factor:.2f}x")
            with col3:
                st.metric("Georeference", "✅ Available" if has_geo else "❌ Missing")
            
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
    
    # Interactive image display
    if st.session_state.geo_locator is not None and st.session_state.display_image is not None:
        st.subheader("Interactive Map")
        st.write("Click anywhere on the image to see GPS coordinates at that location.")
        
        # Create two columns: image and coordinates
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display the image with click coordinates
            from streamlit_image_coordinates import streamlit_image_coordinates
            
            # Convert numpy array to PIL Image for streamlit_image_coordinates
            if isinstance(st.session_state.display_image, np.ndarray):
                display_pil = Image.fromarray(st.session_state.display_image)
            else:
                display_pil = st.session_state.display_image
            
            # Get click coordinates
            coords = streamlit_image_coordinates(
                display_pil,
                key="image_coords",
                width=display_pil.width if hasattr(display_pil, 'width') else st.session_state.display_image.shape[1]
            )
        
        with col2:
            st.subheader("📍 Location Info")
            
            if coords is not None and 'x' in coords and 'y' in coords:
                # Convert display coordinates to original image coordinates
                original_x = coords['x'] / st.session_state.scale_factor
                original_y = coords['y'] / st.session_state.scale_factor
                
                try:
                    # Get GPS coordinates
                    longitude, latitude = st.session_state.geo_locator.pixel_to_gps(original_x, original_y)
                    
                    # Display coordinates
                    st.metric("Pixel X", f"{int(original_x)}")
                    st.metric("Pixel Y", f"{int(original_y)}")
                    st.metric("Longitude", f"{longitude:.6f}°")
                    st.metric("Latitude", f"{latitude:.6f}°")
                    
                    # Google Maps link
                    maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"
                    st.markdown(f"🗺️ [View on Google Maps]({maps_url})")
                    
                    # Coordinates in different formats
                    with st.expander("📋 Copy Coordinates"):
                        st.code(f"Decimal: {latitude:.6f}, {longitude:.6f}")
                        st.code(f"DMS: {dd_to_dms(latitude, longitude)}")
                        st.code(f"UTM: {dd_to_utm(latitude, longitude)}")
                        
                except Exception as e:
                    st.error(f"Error calculating GPS coordinates: {e}")
            else:
                st.info("👆 Click on the image to see GPS coordinates")
    
    else:
        st.info("📁 Please upload a GeoTIFF file to begin.")

def dd_to_dms(lat, lon):
    """Convert decimal degrees to degrees, minutes, seconds"""
    def convert(coord, is_lat=True):
        abs_coord = abs(coord)
        deg = int(abs_coord)
        min_val = int((abs_coord - deg) * 60)
        sec = ((abs_coord - deg) * 60 - min_val) * 60
        
        if is_lat:
            direction = 'N' if coord >= 0 else 'S'
        else:
            direction = 'E' if coord >= 0 else 'W'
        
        return f"{deg}°{min_val}'{sec:.2f}\"{direction}"
    
    lat_dms = convert(lat, True)
    lon_dms = convert(lon, False)
    return f"{lat_dms}, {lon_dms}"

def dd_to_utm(lat, lon):
    """Simple UTM zone calculation (approximate)"""
    zone = int((lon + 180) / 6) + 1
    hemisphere = "N" if lat >= 0 else "S"
    return f"Zone {zone}{hemisphere} (approximate)"

if __name__ == "__main__":
    # Check if required package is installed
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except ImportError:
        st.error("Please install streamlit-image-coordinates: `pip install streamlit-image-coordinates`")
        st.stop()
    
    main()