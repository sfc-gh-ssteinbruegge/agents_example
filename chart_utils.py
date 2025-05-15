"""
Chart Utilities
==============
Shared chart creation utilities for Cortex Analyst and Report Designer.
This module provides centralized chart generation logic to ensure consistent
visualization between different parts of the application.
"""
import pandas as pd
import altair as alt
import streamlit as st
import pydeck as pdk


def create_chart_from_metadata(df):
    """
    Create an Altair chart based on the chart_metadata in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with chart_metadata attribute containing chart configuration
        
    Returns:
    --------
    altair.Chart or pydeck.Deck or None
        The created chart object or None if chart couldn't be created
    """
    try:
        if not hasattr(df, 'attrs') or 'chart_metadata' not in df.attrs:
            return None
            
        chart_metadata = df.attrs.get('chart_metadata', {})
        
        # Determine which chart type to create based on metadata
        if 'chart10_columns' in chart_metadata:
            return create_chart10(df, chart_metadata['chart10_columns'])
        elif 'chart1_columns' in chart_metadata:
            return create_chart1(df, chart_metadata['chart1_columns'])
        elif 'chart2_columns' in chart_metadata:
            return create_chart2(df, chart_metadata['chart2_columns'])
        elif 'chart3_columns' in chart_metadata:
            return create_chart3(df, chart_metadata['chart3_columns'])
        elif 'chart4_columns' in chart_metadata:
            return create_chart4(df, chart_metadata['chart4_columns'])
        elif 'chart5_columns' in chart_metadata:
            return create_chart5(df, chart_metadata['chart5_columns'])
        elif 'chart6_columns' in chart_metadata:
            return create_chart6(df, chart_metadata['chart6_columns'])
        elif 'chart7_columns' in chart_metadata:
            return create_chart7(df, chart_metadata['chart7_columns'])
        elif 'chart8_columns' in chart_metadata:
            return create_chart8(df, chart_metadata['chart8_columns'])
        elif 'chart9_columns' in chart_metadata:
            return create_chart9(df, chart_metadata['chart9_columns'])
        
        # If no specific metadata found, return None
        return None
        
    except Exception as e:
        print(f"Error creating chart from metadata: {str(e)}")
        return None


def create_chart1(df, cols):
    """
    Create Chart 1: Bar Chart by Date
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    cols : dict
        Column configuration with date_col and numeric_col
        
    Returns:
    --------
    altair.Chart
        Bar chart with date on x-axis and numeric value on y-axis
    """
    try:
        date_col = cols.get('date_col')
        numeric_col = cols.get('numeric_col')
        
        return alt.Chart(df).mark_bar().encode(
            x=alt.X(date_col + ':T', sort='ascending'),
            y=alt.Y(numeric_col + ':Q'),
            tooltip=[date_col, numeric_col]
        ).properties(title='1 Bar Chart by Date')
    except Exception as e:
        print(f"Error creating Chart 1: {str(e)}")
        return None


def create_chart2(df, cols):
    """
    Create Chart 2: Dual Axis Line Chart
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    cols : dict
        Column configuration with date_col, num_col1, and num_col2
        
    Returns:
    --------
    altair.LayerChart
        Dual line chart with date on x-axis and two numeric values on independent y-axes
    """
    try:
        date_col = cols.get('date_col')
        num_col1 = cols.get('num_col1')
        num_col2 = cols.get('num_col2')
        
        base = alt.Chart(df).encode(x=alt.X(date_col + ':T', sort='ascending'))
        line1 = base.mark_line(color='blue').encode(
            y=alt.Y(num_col1 + ':Q', axis=alt.Axis(title=num_col1)),
            tooltip=[date_col, num_col1]
        )
        line2 = base.mark_line(color='red').encode(
            y=alt.Y(num_col2 + ':Q', axis=alt.Axis(title=num_col2)),
            tooltip=[date_col, num_col2]
        )
        return alt.layer(line1, line2).resolve_scale(
            y='independent'
        ).properties(title='2 Dual Axis Line Chart')
    except Exception as e:
        print(f"Error creating Chart 2: {str(e)}")
        return None


def create_chart3(df, cols):
    """
    Create Chart 3: Stacked Bar Chart by Date
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    cols : dict
        Column configuration with date_col, text_col, and numeric_col
        
    Returns:
    --------
    altair.Chart
        Stacked bar chart with date on x-axis, numeric value on y-axis, and categorical color
    """
    try:
        date_col = cols.get('date_col')
        text_col = cols.get('text_col')
        numeric_col = cols.get('numeric_col')
        
        return alt.Chart(df).mark_bar().encode(
            x=alt.X(date_col + ':T', sort='ascending'),
            y=alt.Y(numeric_col + ':Q', stack='zero'),
            color=alt.Color(text_col + ':N'),
            tooltip=[date_col, text_col, numeric_col]
        ).properties(title='3 Stacked Bar Chart by Date')
    except Exception as e:
        print(f"Error creating Chart 3: {str(e)}")
        return None


def create_chart4(df, cols):
    """
    Create Chart 4: Stacked Bar Chart with Text Column Selector for Colors
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    cols : dict
        Column configuration with date_col, text_cols, and numeric_col
        
    Returns:
    --------
    altair.Chart
        Stacked bar chart with date on x-axis, numeric value on y-axis, and selectable categorical colors
    """
    try:
        date_col = cols.get('date_col')
        text_cols = cols.get('text_cols', [])
        numeric_col = cols.get('numeric_col')
        
        # Ensure we have at least one text column
        if not text_cols:
            # Find suitable text columns if not specified
            all_cols = list(df.columns)
            possible_text_cols = [col for col in all_cols if col != date_col and col != numeric_col]
            if possible_text_cols:
                text_cols = possible_text_cols
            else:
                # If no text columns available, return None
                return None
        
        # Generate a unique key for this chart based on dataframe and columns
        df_hash = hash(str(df.shape) + str(df.columns.tolist()))
        widget_key = f"chart4_select_{df_hash}"
        
        # Initialize the session state value if it doesn't exist
        if widget_key not in st.session_state:
            st.session_state[widget_key] = text_cols[0]
        # If the value exists but is not in text_cols (changed data), reset it
        elif st.session_state[widget_key] not in text_cols:
            st.session_state[widget_key] = text_cols[0]
        
        # Get the selected column from session state
        selected_text_col = st.selectbox(
            "Select column for color grouping:",
            options=text_cols,
            index=text_cols.index(st.session_state[widget_key]),
            key=widget_key
        )
        
        # Create the chart with the selected text column
        return alt.Chart(df).mark_bar().encode(
            x=alt.X(date_col + ':T', sort='ascending'),
            y=alt.Y(numeric_col + ':Q', stack='zero'),
            color=alt.Color(selected_text_col + ':N', title=selected_text_col),
            tooltip=[date_col, selected_text_col, numeric_col]
        ).properties(title='4 Stacked Bar Chart with Selectable Colors')
    except Exception as e:
        print(f"Error creating Chart 4: {str(e)}")
        return None


def create_chart5(df, cols):
    """
    Create Chart 5: Scatter Chart
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    cols : dict
        Column configuration with num_col1, num_col2, and text_col
        
    Returns:
    --------
    altair.Chart
        Scatter chart with numeric x/y and categorical color
    """
    try:
        num_col1 = cols.get('num_col1')
        num_col2 = cols.get('num_col2')
        text_col = cols.get('text_col')
        
        return alt.Chart(df).mark_circle(size=100).encode(
            x=alt.X(num_col1 + ':Q'),
            y=alt.Y(num_col2 + ':Q'),
            color=alt.Color(text_col + ':N'),
            tooltip=[text_col, num_col1, num_col2]
        ).properties(title='5 Scatter Chart')
    except Exception as e:
        print(f"Error creating Chart 5: {str(e)}")
        return None


def create_chart6(df, cols):
    """
    Create Chart 6: Scatter Chart with Multiple Dimensions
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    cols : dict
        Column configuration with num_col1, num_col2, text_col1, and text_col2
        
    Returns:
    --------
    altair.Chart
        Scatter chart with numeric x/y and categorical color and shape
    """
    try:
        num_col1 = cols.get('num_col1')
        num_col2 = cols.get('num_col2')
        text_col1 = cols.get('text_col1')
        text_col2 = cols.get('text_col2')
        
        return alt.Chart(df).mark_point(size=100).encode(
            x=alt.X(num_col1 + ':Q'),
            y=alt.Y(num_col2 + ':Q'),
            color=alt.Color(text_col1 + ':N'),
            shape=alt.Shape(text_col2 + ':N', scale=alt.Scale(
                range=["circle", "square", "cross", "diamond", "triangle-up", "triangle-down", 
                       "triangle-right", "triangle-left", "arrow", "wedge", "stroke"]
            )),
            tooltip=[text_col1, text_col2, num_col1, num_col2]
        ).properties(title='6 Scatter Chart with Multiple Dimensions')
    except Exception as e:
        print(f"Error creating Chart 6: {str(e)}")
        return None


def create_chart7(df, cols):
    """
    Create Chart 7: Bubble Chart
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    cols : dict
        Column configuration with num_col1, num_col2, num_col3, and text_col
        
    Returns:
    --------
    altair.Chart
        Bubble chart with numeric x/y/size and categorical color
    """
    try:
        num_col1 = cols.get('num_col1')
        num_col2 = cols.get('num_col2')
        num_col3 = cols.get('num_col3')
        text_col = cols.get('text_col')
        
        return alt.Chart(df).mark_circle().encode(
            x=alt.X(num_col1 + ':Q'),
            y=alt.Y(num_col2 + ':Q'),
            size=alt.Size(num_col3 + ':Q'),
            color=alt.Color(text_col + ':N'),
            tooltip=[text_col, num_col1, num_col2, num_col3]
        ).properties(title='7 Bubble Chart')
    except Exception as e:
        print(f"Error creating Chart 7: {str(e)}")
        return None


def create_chart8(df, cols):
    """
    Create Chart 8: Multi-Dimensional Bubble Chart
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    cols : dict
        Column configuration with num_col1, num_col2, num_col3, text_col1, and text_col2
        
    Returns:
    --------
    altair.Chart
        Multi-dimensional bubble chart with numeric x/y/size and categorical color and shape
    """
    try:
        num_col1 = cols.get('num_col1')
        num_col2 = cols.get('num_col2')
        num_col3 = cols.get('num_col3')
        text_col1 = cols.get('text_col1')
        text_col2 = cols.get('text_col2')
        
        return alt.Chart(df).mark_point().encode(
            x=alt.X(num_col1 + ':Q'),
            y=alt.Y(num_col2 + ':Q'),
            size=alt.Size(num_col3 + ':Q'),
            color=alt.Color(text_col1 + ':N'),
            shape=alt.Shape(text_col2 + ':N', scale=alt.Scale(
                range=["circle", "square", "cross", "diamond", "triangle-up", "triangle-down", 
                       "triangle-right", "triangle-left", "arrow", "wedge", "stroke"]
            )),
            tooltip=[text_col1, text_col2, num_col1, num_col2, num_col3]
        ).properties(title='8 Multi-Dimensional Bubble Chart')
    except Exception as e:
        print(f"Error creating Chart 8: {str(e)}")
        return None


def create_chart9(df, cols):
    """
    Create Chart 9: Bar Chart with Text Column Selector
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data
    cols : dict
        Column configuration with numeric_col and text_cols
        
    Returns:
    --------
    altair.Chart and selectbox widget
        Bar chart with numeric value on y-axis and selected text column on x-axis
    """
    try:
        numeric_col = cols.get('numeric_col')
        text_cols = cols.get('text_cols', [])
        
        if not text_cols:
            return None
        
        # Generate a unique key for this chart based on dataframe and columns
        df_hash = hash(str(df.shape) + str(df.columns.tolist()))
        widget_key = f"chart9_select_{df_hash}"
        
        # Initialize the session state value if it doesn't exist
        if widget_key not in st.session_state:
            st.session_state[widget_key] = text_cols[0]
        # If the value exists but is not in text_cols (changed data), reset it
        elif st.session_state[widget_key] not in text_cols:
            st.session_state[widget_key] = text_cols[0]
        
        # Get the selected column from session state
        selected_text_col = st.selectbox(
            "Select column for X-axis:",
            options=text_cols,
            index=text_cols.index(st.session_state[widget_key]),
            key=widget_key
        )
        
        # Create the bar chart with the selected text column
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{selected_text_col}:N", sort='-y'),
            y=alt.Y(f"{numeric_col}:Q"),
            tooltip=[selected_text_col, numeric_col]
        ).properties(title='9 Bar Chart with Selectable X-Axis')
        
        return chart
    except Exception as e:
        print(f"Error creating Chart 9: {str(e)}")
        return None


def create_chart10(df, cols):
    """
    Create Chart 10: Map Visualization with PyDeck
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data containing latitude and longitude
    cols : dict
        Column configuration with lat_col, lon_col, and optionally color_col and size_col
        
    Returns:
    --------
    pydeck.Deck
        PyDeck map visualization with points plotted
    """
    try:
        lat_col = cols.get('lat_col')
        lon_col = cols.get('lon_col')
        color_col = cols.get('color_col')
        size_col = cols.get('size_col')
        
        if not lat_col or not lon_col:
            return None
            
        # Calculate center point for initial view state
        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()
        
        # Prepare data for visualization
        data = {
            'latitude': df[lat_col],
            'longitude': df[lon_col]
        }
        
        # Add color and size if specified
        if color_col and color_col in df.columns:
            data['color'] = df[color_col]
            # If color column is categorical, create a color mapping
            if not pd.api.types.is_numeric_dtype(df[color_col]):
                unique_categories = df[color_col].unique()
                color_scale = {
                    cat: [
                        int(255 * (i / len(unique_categories))),
                        100,
                        int(255 * (1 - i / len(unique_categories))),
                        200
                    ] for i, cat in enumerate(unique_categories)
                }
                data['color'] = df[color_col].map(lambda x: color_scale.get(x, [255, 140, 0, 100]))
            else:
                # Normalize numeric color values to 0-255 range
                min_val = df[color_col].min()
                max_val = df[color_col].max()
                if min_val != max_val:
                    data['color'] = df[color_col].apply(lambda x: [
                        int(255 * (x - min_val) / (max_val - min_val)),
                        140,
                        int(255 * (1 - (x - min_val) / (max_val - min_val))),
                        100
                    ])
                else:
                    data['color'] = [[255, 140, 0, 100]] * len(df)
        
        if size_col and size_col in df.columns:
            data['size'] = df[size_col]
            # Normalize size values between 100 and 1000
            min_size = data['size'].min()
            max_size = data['size'].max()
            if min_size != max_size:
                data['size'] = 100 + 900 * (data['size'] - min_size) / (max_size - min_size)
            else:
                data['size'] = 300  # Default size if all values are the same
        
        map_data = pd.DataFrame(data)
        
        # Define the layer
        layer_props = {
            "data": map_data,
            "get_position": "[longitude, latitude]",
            "get_radius": 300 if not size_col else "size",
            "get_fill_color": [255, 140, 0, 100] if not color_col else "color",
            "pickable": True,
            "opacity": 0.8,
            "stroked": True,
            "filled": True,
            "radius_scale": 3,
            "radius_min_pixels": 2,
            "radius_max_pixels": 15,
        }
        
        scatterplot_layer = pdk.Layer("ScatterplotLayer", **layer_props)
        
        # Create the deck with US-focused initial view
        deck = pdk.Deck(
            layers=[scatterplot_layer],
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=11 if abs(center_lat) < 50 and -125 < center_lon < -65 else 4,  # Zoom in more for US locations
                pitch=0,
            ),
            map_style="mapbox://styles/mapbox/light-v9",
            tooltip={
                "html": "<b>Latitude:</b> {{latitude}}<br/>"
                       "<b>Longitude:</b> {{longitude}}<br/>"
                       + (f"<b>{color_col}:</b> {{{{color}}}}<br/>" if color_col and not isinstance(data.get('color', None), list) else "")
                       + (f"<b>{size_col}:</b> {{{{size}}}}<br/>" if size_col else ""),
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
        )
        
        return deck
        
    except Exception as e:
        print(f"Error creating Chart 10: {str(e)}")
        return None


# Utility functions for common chart operations
def detect_column_types(df):
    """
    Automatically detect different column types in a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    dict
        Dictionary with categorized columns (date_cols, numeric_cols, text_cols, lat_cols, lon_cols)
    """
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Enhanced geographic data detection
    lat_cols = []
    lon_cols = []
    for col in numeric_cols:
        col_lower = col.lower()
        # Check for latitude columns
        if any(lat_term in col_lower for lat_term in ['lat', 'latitude']):
            # Validate latitude range
            if df[col].min() >= -90 and df[col].max() <= 90:
                lat_cols.append(col)
        # Check for longitude columns
        elif any(lon_term in col_lower for lon_term in ['lon', 'long', 'longitude']):
            # Validate longitude range
            if df[col].min() >= -180 and df[col].max() <= 180:
                lon_cols.append(col)
    
    # Remove lat/lon columns from numeric_cols to avoid double counting
    numeric_cols = [col for col in numeric_cols if col not in lat_cols and col not in lon_cols]
    text_cols = [col for col in df.columns if col not in numeric_cols + date_cols + lat_cols + lon_cols]
    
    return {
        'date_cols': date_cols,
        'numeric_cols': numeric_cols,
        'text_cols': text_cols,
        'lat_cols': lat_cols,
        'lon_cols': lon_cols
    }


def suggest_chart_type(df):
    """
    Analyze a DataFrame and suggest an appropriate chart type
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    str
        Suggested chart type based on data structure
    """
    col_types = detect_column_types(df)
    date_cols = col_types['date_cols']
    numeric_cols = col_types['numeric_cols']
    text_cols = col_types['text_cols']
    lat_cols = col_types['lat_cols']
    lon_cols = col_types['lon_cols']
    
    # Chart 10: Check for latitude and longitude columns first
    if len(lat_cols) >= 1 and len(lon_cols) >= 1:
        return 'chart10'
    
    # Chart 1: Single date column, single numeric column
    elif len(date_cols) == 1 and len(numeric_cols) == 1 and len(text_cols) == 0:
        return 'chart1'
        
    # Chart 2: Single date column, multiple numeric columns
    elif len(date_cols) == 1 and len(numeric_cols) >= 2 and len(text_cols) == 0:
        return 'chart2'
        
    # Chart 3: Date column, numeric column, and one categorical column
    elif len(date_cols) == 1 and len(numeric_cols) >= 1 and len(text_cols) == 1:
        return 'chart3'
        
    # Chart 4: Date column, numeric column, and multiple categorical columns
    elif len(date_cols) == 1 and len(numeric_cols) >= 1 and len(text_cols) >= 2:
        return 'chart4'
        
    # Default to generic bar chart
    return 'bar'


def generate_chart_code_for_dataframe(df):
    """
    Generate chart code for a dataframe based on its chart_metadata.
    This centralizes chart code generation to avoid duplication across pages.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with chart_metadata attribute containing chart configuration
        
    Returns:
    --------
    str
        The generated chart code as a string
    """
    import io
    buf = io.StringIO()
    print("import altair as alt", file=buf)
    print("import pandas as pd", file=buf)
    print("\n# Chart code", file=buf)
    print("def create_chart(df):", file=buf)
    
    # Determine which chart type we have based on metadata and generate appropriate code
    if hasattr(df, 'attrs') and 'chart_metadata' in df.attrs:
        chart_metadata = df.attrs['chart_metadata']
        
        if 'chart1_columns' in chart_metadata:
            cols = chart_metadata['chart1_columns']
            date_col = cols.get('date_col')
            numeric_col = cols.get('numeric_col')
            
            if not date_col or not numeric_col:
                print(f"    # Error: Missing required columns for chart1", file=buf)
                print(f"    st.error('Missing required columns for Bar Chart by Date')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    return alt.Chart(df).mark_bar().encode(", file=buf)
                print(f"        x=alt.X('{date_col}:T', sort='ascending'),", file=buf)
                print(f"        y=alt.Y('{numeric_col}:Q'),", file=buf)
                print(f"        tooltip=['{date_col}', '{numeric_col}']", file=buf)
                print(f"    ).properties(title='Bar Chart by Date')", file=buf)
            
        elif 'chart2_columns' in chart_metadata:
            cols = chart_metadata['chart2_columns']
            date_col = cols.get('date_col')
            num_col1 = cols.get('num_col1')
            num_col2 = cols.get('num_col2')
            
            if not date_col or not num_col1 or not num_col2:
                print(f"    # Error: Missing required columns for chart2", file=buf)
                print(f"    st.error('Missing required columns for Dual Axis Line Chart')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    base = alt.Chart(df).encode(x=alt.X('{date_col}:T', sort='ascending'))", file=buf)
                print(f"    line1 = base.mark_line(color='blue').encode(", file=buf)
                print(f"        y=alt.Y('{num_col1}:Q', axis=alt.Axis(title='{num_col1}')),", file=buf)
                print(f"        tooltip=['{date_col}', '{num_col1}']", file=buf)
                print(f"    )", file=buf)
                print(f"    line2 = base.mark_line(color='red').encode(", file=buf)
                print(f"        y=alt.Y('{num_col2}:Q', axis=alt.Axis(title='{num_col2}')),", file=buf)
                print(f"        tooltip=['{date_col}', '{num_col2}']", file=buf)
                print(f"    )", file=buf)
                print(f"    return alt.layer(line1, line2).resolve_scale(", file=buf)
                print(f"        y='independent'", file=buf)
                print(f"    ).properties(title='Dual Axis Line Chart')", file=buf)
        
        elif 'chart3_columns' in chart_metadata:
            cols = chart_metadata['chart3_columns']
            date_col = cols.get('date_col')
            text_col = cols.get('text_col')
            numeric_col = cols.get('numeric_col')
            
            if not date_col or not text_col or not numeric_col:
                print(f"    # Error: Missing required columns for chart3", file=buf)
                print(f"    st.error('Missing required columns for Stacked Bar Chart by Date')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    return alt.Chart(df).mark_bar().encode(", file=buf)
                print(f"        x=alt.X('{date_col}:T', sort='ascending'),", file=buf)
                print(f"        y=alt.Y('{numeric_col}:Q', stack='zero'),", file=buf)
                print(f"        color=alt.Color('{text_col}:N'),", file=buf)
                print(f"        tooltip=['{date_col}', '{text_col}', '{numeric_col}']", file=buf)
                print(f"    ).properties(title='Stacked Bar Chart by Date')", file=buf)
        
        elif 'chart4_columns' in chart_metadata:
            cols = chart_metadata['chart4_columns']
            date_col = cols.get('date_col')
            text_cols = cols.get('text_cols', [])
            numeric_col = cols.get('numeric_col')
            
            if not date_col or not text_cols or not numeric_col or len(text_cols) < 2:
                print(f"    # Error: Missing required columns for chart4", file=buf)
                print(f"    st.error('Missing required columns for Stacked Bar Chart with Text Column Selector for Colors')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    # Generate a unique key for this chart based on dataframe and columns", file=buf)
                print(f"    df_hash = hash(str(df.shape) + str(df.columns.tolist()))", file=buf)
                print(f"    widget_key = f\"chart4_select_{{df_hash}}\"", file=buf)
                print(f"", file=buf)
                print(f"    # Initialize the session state value if it doesn't exist", file=buf)
                print(f"    if widget_key not in st.session_state:", file=buf)
                print(f"        st.session_state[widget_key] = {text_cols}[0]", file=buf)
                print(f"    # If the value exists but is not in text_cols (changed data), reset it", file=buf)
                print(f"    elif st.session_state[widget_key] not in {text_cols}:", file=buf)
                print(f"        st.session_state[widget_key] = {text_cols}[0]", file=buf)
                print(f"", file=buf)
                print(f"    # Get the selected column from session state", file=buf)
                print(f"    selected_text_col = st.selectbox(", file=buf)
                print(f"        \"Select column for color grouping:\",", file=buf)
                print(f"        options={text_cols},", file=buf)
                print(f"        index={text_cols}.index(st.session_state[widget_key]),", file=buf)
                print(f"        key=widget_key", file=buf)
                print(f"    )", file=buf)
                print(f"", file=buf)
                print(f"    # Create the chart with the selected text column", file=buf)
                print(f"    return alt.Chart(df).mark_bar().encode(", file=buf)
                print(f"        x=alt.X('{date_col}:T', sort='ascending'),", file=buf)
                print(f"        y=alt.Y('{numeric_col}:Q', stack='zero'),", file=buf)
                print(f"        color=alt.Color('{{selected_text_col}}:N', title='{{selected_text_col}}'),", file=buf)
                print(f"        tooltip=['{date_col}', '{{selected_text_col}}', '{numeric_col}']", file=buf)
                print(f"    ).properties(title='Stacked Bar Chart with Selectable Colors')", file=buf)
        
        elif 'chart5_columns' in chart_metadata:
            cols = chart_metadata['chart5_columns']
            num_col1 = cols.get('num_col1')
            num_col2 = cols.get('num_col2')
            text_col = cols.get('text_col')
            
            if not num_col1 or not num_col2 or not text_col:
                print(f"    # Error: Missing required columns for chart5", file=buf)
                print(f"    st.error('Missing required columns for Scatter Chart')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    return alt.Chart(df).mark_circle(size=60).encode(", file=buf)
                print(f"        x=alt.X('{num_col1}:Q'),", file=buf)
                print(f"        y=alt.Y('{num_col2}:Q'),", file=buf)
                print(f"        color=alt.Color('{text_col}:N'),", file=buf)
                print(f"        tooltip=['{num_col1}', '{num_col2}', '{text_col}']", file=buf)
                print(f"    ).properties(title='Scatter Plot')", file=buf)
        
        elif 'chart6_columns' in chart_metadata:
            cols = chart_metadata['chart6_columns']
            num_col1 = cols.get('num_col1')
            num_col2 = cols.get('num_col2')
            text_col1 = cols.get('text_col1')
            text_col2 = cols.get('text_col2')
            
            if not num_col1 or not num_col2 or not text_col1 or not text_col2:
                print(f"    # Error: Missing required columns for chart6", file=buf)
                print(f"    st.error('Missing required columns for Scatter Chart with Multiple Dimensions')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    return alt.Chart(df).mark_point(size=100).encode(", file=buf)
                print(f"        x=alt.X('{num_col1}:Q'),", file=buf)
                print(f"        y=alt.Y('{num_col2}:Q'),", file=buf)
                print(f"        color=alt.Color('{text_col1}:N'),", file=buf)
                print(f"        shape=alt.Shape('{text_col2}:N', scale=alt.Scale(", file=buf)
                print(f"            range=[\"circle\", \"square\", \"cross\", \"diamond\", \"triangle-up\", \"triangle-down\", ", file=buf)
                print(f"                   \"triangle-right\", \"triangle-left\", \"arrow\", \"wedge\", \"stroke\"]", file=buf)
                print(f"        )),", file=buf)
                print(f"        tooltip=['{text_col1}', '{text_col2}', '{num_col1}', '{num_col2}']", file=buf)
                print(f"    ).properties(title='Scatter Chart with Multiple Dimensions')", file=buf)
        
        elif 'chart7_columns' in chart_metadata:
            cols = chart_metadata['chart7_columns']
            num_col1 = cols.get('num_col1')
            num_col2 = cols.get('num_col2')
            num_col3 = cols.get('num_col3')
            text_col = cols.get('text_col')
            
            if not num_col1 or not num_col2 or not num_col3 or not text_col:
                print(f"    # Error: Missing required columns for chart7", file=buf)
                print(f"    st.error('Missing required columns for Bubble Chart')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    return alt.Chart(df).mark_circle().encode(", file=buf)
                print(f"        x=alt.X('{num_col1}:Q'),", file=buf)
                print(f"        y=alt.Y('{num_col2}:Q'),", file=buf)
                print(f"        size=alt.Size('{num_col3}:Q'),", file=buf)
                print(f"        color=alt.Color('{text_col}:N'),", file=buf)
                print(f"        tooltip=['{num_col1}', '{num_col2}', '{num_col3}', '{text_col}']", file=buf)
                print(f"    ).properties(title='Bubble Chart')", file=buf)
        
        elif 'chart8_columns' in chart_metadata:
            cols = chart_metadata['chart8_columns']
            num_col1 = cols.get('num_col1')
            num_col2 = cols.get('num_col2')
            num_col3 = cols.get('num_col3')
            text_col1 = cols.get('text_col1')
            text_col2 = cols.get('text_col2')
            
            if not num_col1 or not num_col2 or not num_col3 or not text_col1 or not text_col2:
                print(f"    # Error: Missing required columns for chart8", file=buf)
                print(f"    st.error('Missing required columns for Multi-Dimensional Bubble Chart')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    return alt.Chart(df).mark_point().encode(", file=buf)
                print(f"        x=alt.X('{num_col1}:Q'),", file=buf)
                print(f"        y=alt.Y('{num_col2}:Q'),", file=buf)
                print(f"        size=alt.Size('{num_col3}:Q'),", file=buf)
                print(f"        color=alt.Color('{text_col1}:N'),", file=buf)
                print(f"        shape=alt.Shape('{text_col2}:N', scale=alt.Scale(", file=buf)
                print(f"            range=[\"circle\", \"square\", \"cross\", \"diamond\", \"triangle-up\", \"triangle-down\", ", file=buf)
                print(f"                   \"triangle-right\", \"triangle-left\", \"arrow\", \"wedge\", \"stroke\"]", file=buf)
                print(f"        )),", file=buf)
                print(f"        tooltip=['{text_col1}', '{text_col2}', '{num_col1}', '{num_col2}', '{num_col3}']", file=buf)
                print(f"    ).properties(title='Multi-Dimensional Bubble Chart')", file=buf)
        
        elif 'chart9_columns' in chart_metadata:
            cols = chart_metadata['chart9_columns']
            numeric_col = cols.get('numeric_col')
            text_cols = cols.get('text_cols')
            
            if not numeric_col or not text_cols or len(text_cols) == 0:
                print(f"    # Error: Missing required columns for chart9", file=buf)
                print(f"    st.error('Missing required columns for Bar Chart with Text Column Selector')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    # Generate a unique key for this chart based on dataframe and columns", file=buf)
                print(f"    df_hash = hash(str(df.shape) + str(df.columns.tolist()))", file=buf)
                print(f"    widget_key = f\"chart9_select_{{df_hash}}\"", file=buf)
                print(f"", file=buf)
                print(f"    # Initialize the session state value if it doesn't exist", file=buf)
                print(f"    if widget_key not in st.session_state:", file=buf)
                print(f"        st.session_state[widget_key] = {text_cols}[0]", file=buf)
                print(f"    # If the value exists but is not in text_cols (changed data), reset it", file=buf)
                print(f"    elif st.session_state[widget_key] not in {text_cols}:", file=buf)
                print(f"        st.session_state[widget_key] = {text_cols}[0]", file=buf)
                print(f"", file=buf)
                print(f"    # Get the selected column from session state", file=buf)
                print(f"    selected_text_col = st.selectbox(", file=buf)
                print(f"        \"Select column for X-axis:\",", file=buf)
                print(f"        options={text_cols},", file=buf)
                print(f"        index={text_cols}.index(st.session_state[widget_key]),", file=buf)
                print(f"        key=widget_key", file=buf)
                print(f"    )", file=buf)
                print(f"", file=buf)
                print(f"    # Create the bar chart with the selected text column", file=buf)
                print(f"    return alt.Chart(df).mark_bar().encode(", file=buf)
                print(f"        x=alt.X(f\"{{selected_text_col}}:N\", sort='-y'),", file=buf)
                print(f"        y=alt.Y(f\"{numeric_col}:Q\"),", file=buf)
                print(f"        tooltip=[selected_text_col, '{numeric_col}']", file=buf)
                print(f"    ).properties(title='Bar Chart with Selectable X-Axis')", file=buf)
        
        elif 'chart10_columns' in chart_metadata:
            cols = chart_metadata['chart10_columns']
            lat_col = cols.get('lat_col')
            lon_col = cols.get('lon_col')
            color_col = cols.get('color_col')
            size_col = cols.get('size_col')
            
            if not lat_col or not lon_col:
                print(f"    # Error: Missing required columns for chart10", file=buf)
                print(f"    st.error('Missing required columns for Map Visualization with PyDeck')", file=buf)
                print(f"    return None", file=buf)
            else:
                print(f"    # Calculate center point for initial view state", file=buf)
                print(f"    center_lat = df[lat_col].mean()", file=buf)
                print(f"    center_lon = df[lon_col].mean()", file=buf)
                print(f"", file=buf)
                print(f"    # Prepare data for visualization", file=buf)
                print(f"    data = {{", file=buf)
                print(f"        'latitude': df[lat_col],", file=buf)
                print(f"        'longitude': df[lon_col]", file=buf)
                print(f"    }}", file=buf)
                print(f"", file=buf)
                print(f"    # Add color and size if specified", file=buf)
                print(f"    if color_col and color_col in df.columns:", file=buf)
                print(f"        data['color'] = df[color_col]", file=buf)
                print(f"    if size_col and size_col in df.columns:", file=buf)
                print(f"        data['size'] = df[size_col]", file=buf)
                print(f"        # Normalize size values between 100 and 1000", file=buf)
                print(f"        min_size = data['size'].min()", file=buf)
                print(f"        max_size = data['size'].max()", file=buf)
                print(f"        if min_size != max_size:", file=buf)
                print(f"            data['size'] = 100 + 900 * (data['size'] - min_size) / (max_size - min_size)", file=buf)
                print(f"        else:", file=buf)
                print(f"            data['size'] = 300  # Default size if all values are the same", file=buf)
                print(f"", file=buf)
                print(f"    map_data = pd.DataFrame(data)", file=buf)
                print(f"", file=buf)
                print(f"    # Define the layer", file=buf)
                print(f"    layer_props = {{", file=buf)
                print(f"        \"data\": map_data,", file=buf)
                print(f"        \"get_position\": f\"[longitude, latitude]\",", file=buf)
                print(f"        \"get_radius\": 300 if not size_col else \"size\",", file=buf)
                print(f"        \"get_fill_color\": [255, 140, 0, 100] if not color_col else \"color\",", file=buf)
                print(f"        \"pickable\": True,", file=buf)
                print(f"        \"opacity\": 0.8,", file=buf)
                print(f"        \"stroked\": True,", file=buf)
                print(f"        \"filled\": True,", file=buf)
                print(f"        \"radius_scale\": 3,", file=buf)
                print(f"        \"radius_min_pixels\": 2,", file=buf)
                print(f"        \"radius_max_pixels\": 15,", file=buf)
                print(f"    }}", file=buf)
                print(f"", file=buf)
                print(f"    scatterplot_layer = pdk.Layer(\"ScatterplotLayer\", **layer_props)", file=buf)
                print(f"", file=buf)
                print(f"    # Create the deck", file=buf)
                print(f"    deck = pdk.Deck(", file=buf)
                print(f"        layers=[scatterplot_layer],", file=buf)
                print(f"        initial_view_state=pdk.ViewState(", file=buf)
                print(f"            latitude=center_lat,", file=buf)
                print(f"            longitude=center_lon,", file=buf)
                print(f"            zoom=11 if abs(center_lat) < 50 and -125 < center_lon < -65 else 4,  # Zoom in more for US locations", file=buf)
                print(f"            pitch=0,", file=buf)
                print(f"        ),", file=buf)
                print(f"        map_style=\"mapbox://styles/mapbox/light-v9\",", file=buf)
                print(f"        tooltip={{", file=buf)
                print(f"            \"html\": \"<b>Latitude:</b> {{latitude}}<br/>\"", file=buf)
                print(f"               \"<b>Longitude:</b> {{longitude}}<br/>\"", file=buf)
                print(f"               + (f\"<b>{color_col}:</b> {{{{color}}}}<br/>\" if color_col and not isinstance(data.get('color', None), list) else \"\")", file=buf)
                print(f"               + (f\"<b>{size_col}:</b> {{{{size}}}}<br/>\" if size_col else \"\"),", file=buf)
                print(f"            \"style\": {{", file=buf)
                print(f"                \"backgroundColor\": \"steelblue\",", file=buf)
                print(f"                \"color\": \"white\"", file=buf)
                print(f"            }}", file=buf)
                print(f"        )", file=buf)
                print(f"    )", file=buf)
                print(f"", file=buf)
                print(f"    return deck", file=buf)
        
        else:
            # No specific chart type identified in metadata
            print(f"    # No chart type identified in metadata", file=buf)
            print(f"    st.error('No valid chart type found in metadata. Please provide chart configuration.')", file=buf)
            print(f"    return None", file=buf)
    else:
        # No chart metadata
        print(f"    # No chart metadata available", file=buf)
        print(f"    st.error('No chart metadata available. Please provide chart configuration.')", file=buf)
        print(f"    return None", file=buf)
    
    # Return the generated code
    return buf.getvalue() 