# Imports
import streamlit as st
from snowflake.snowpark.session import Session
import requests # For making HTTP requests to the Snowflake API
import os
import json
import copy # To deep copy message list for filtering
import re # For citation and related query handling
import pandas as pd # To handle SQL results
from datetime import datetime # For debug log timestamps
from typing import Generator #, Tuple, List, Any # Import Generator and other types
from sseclient import SSEClient # For parsing Server-Sent Events
import altair as alt # For chart creation
import pydeck as pdk # For PyDeck deck creation

# Import chart utilities
from chart_utils import (
    detect_column_types,
    suggest_chart_type,
    create_chart1,
    create_chart2,
    create_chart3,
    create_chart4,
    create_chart5,
    create_chart6,
    create_chart7,
    create_chart8,
    create_chart9,
    create_chart10
)

# Set page title and icon
st.set_page_config(page_title="Cortex Agent Chat (Standalone)", page_icon="./Q Logo 2024.png", layout="wide")

#  ─── HIDE STREAMLIT UI ─────────────────────────────────────────────────────

st.markdown(
    """
    <style>
      /* hide the "hamburger" menu in the top-right */
      #MainMenu { visibility: hidden !important; }
      /* hide the "Made with Streamlit" footer bar */
      footer { visibility: hidden !important; }
      /* optionally reclaim that footer space */
      .css-18e3th9 { padding-bottom: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
#  ────────────────────────────────────────────────────────────────────────────

SNOWFLAKE_ACCOUNT = st.secrets["SNOWFLAKE_ACCOUNT"] # e.g., xy12345.us-west-2
SNOWFLAKE_USER = st.secrets["SNOWFLAKE_USER"]
SNOWFLAKE_PASSWORD = st.secrets["SNOWFLAKE_PASSWORD"]  # Store password in Streamlit secrets

# --- Other Snowflake Config (Optional but Recommended) ---
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_DATABASE = "TELCO_SAMPLE_DATA"
SNOWFLAKE_SCHEMA = "TELCO_DATASET"
SNOWFLAKE_ROLE = st.secrets["SNOWFLAKE_ROLE"] # Optional: Specify role if needed

# --- API Configuration ---
# Construct the base URL for Snowflake API calls
# Update SNOWFLAKE_ACCOUNT first for this to be correct
SNOWFLAKE_API_BASE_URL = f"https://{SNOWFLAKE_ACCOUNT}.snowflakecomputing.com" if SNOWFLAKE_ACCOUNT != "YOUR_SNOWFLAKE_ACCOUNT_IDENTIFIER" else "https://<YOUR_ACCOUNT>.snowflakecomputing.com"

API_TIMEOUT_SECONDS = 600 # Timeout for requests library (in seconds)
MODEL_NAME = 'claude-3-5-sonnet'

# Citation/Reltaed Queries regex
RELATED_QUERIES_REGEX = r"Related query:\s*(.*?)\s*Answer:\s*(.*?)(?=\nRelated query:|\n*$)"
CITATION_REGEX = r"【†(\d+)†】"

# Keywords for detecting sample requests
SAMPLE_KEYWORDS = ['sample', 'example', 'show me some', 'give me some', 'few examples']

# Define Tools
CORTEX_ANALYST_TOOL_DEF = { "tool_spec": { "type": "cortex_analyst_text_to_sql", "name": "analyst1" } }
CORTEX_SEARCH_TOOL_DEF = { "tool_spec": { "type": "cortex_search", "name": "search1" } }
SQL_EXEC_TOOL_DEF = { "tool_spec": { "type": "sql_exec", "name": "sql_exec" } }

# Define the list of tools to be sent to the API.
AGENT_TOOLS = [
    CORTEX_ANALYST_TOOL_DEF, 
    CORTEX_SEARCH_TOOL_DEF,
    SQL_EXEC_TOOL_DEF
]

# Define Tool Resources, ensure paths/names are valid in your Snowflake account)
# !!! IMPORTANT: Replace placeholder values below !!!
AGENT_TOOL_RESOURCES = {
    "analyst1": { "semantic_model_file": "@TELCO_SAMPLE_DATA.TELCO_DATASET.semantic_models/telco_test_data_complete.yaml" },
    "search1": { "name": "telco_search_service", "max_results": 10 },
}

# Define Experimental Params
AGENT_EXPERIMENTAL_PARAMS = {
    "EnableRelatedQueries": True 
}

# --- Initialize Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "debug_log" not in st.session_state: st.session_state.debug_log = []
if "debug_mode" not in st.session_state: st.session_state.debug_mode = False
if "snowpark_session" not in st.session_state: st.session_state.snowpark_session = None # Store session
if "charts" not in st.session_state: st.session_state.charts = [] # Store chart data
if "visualization_states" not in st.session_state: st.session_state.visualization_states = {} # Store complete visualization states
if "chart_expander_states" not in st.session_state: st.session_state.chart_expander_states = {} # Store expander states
if "table_expander_states" not in st.session_state: st.session_state.table_expander_states = {} # Store table expander states

# --- Helper: Add Log Entry (Conditional) ---
# (No changes needed in this function itself)
def add_log(level, message):
    """Appends a formatted message to the debug log if debug mode is ON."""
    # Check toggle state directly
    if not st.session_state.get("debug_mode", False): return
    # Initialize log if it doesn't exist (shouldn't be needed with init above, but safe)
    if "debug_log" not in st.session_state: st.session_state.debug_log = []
    # Ensure message is string
    if not isinstance(message, str):
        try: message = json.dumps(message, indent=2)
        except TypeError: message = str(message) # Fallback to string conversion
    log_entry = f"{datetime.now()} - {level}: {message}"
    st.session_state.debug_log.append(log_entry)

def create_snowpark_session():
    """Creates and returns a Snowpark session using password authentication."""
    add_log("INFO", "Attempting to create Snowpark session using password authentication...")

    if SNOWFLAKE_ACCOUNT == "YOUR_SNOWFLAKE_ACCOUNT_IDENTIFIER" or SNOWFLAKE_USER == "YOUR_SNOWFLAKE_USER":
        add_log("ERROR", "Cannot create session: Placeholder values for ACCOUNT or USER not replaced.")
        st.sidebar.error("Fatal Error: Snowflake connection details (Account, User) must be set in the script.")
        return None

    try:
        connection_parameters = {
            "account": SNOWFLAKE_ACCOUNT,
            "user": SNOWFLAKE_USER,
            "password": SNOWFLAKE_PASSWORD,
            **({"warehouse": SNOWFLAKE_WAREHOUSE} if SNOWFLAKE_WAREHOUSE else {}),
            **({"database": SNOWFLAKE_DATABASE} if SNOWFLAKE_DATABASE else {}),
            **({"schema": SNOWFLAKE_SCHEMA} if SNOWFLAKE_SCHEMA else {}),
            **({"role": SNOWFLAKE_ROLE} if SNOWFLAKE_ROLE else {})
        }
        
        # Create a loggable version of parameters without sensitive info
        loggable_params = {k: v for k, v in connection_parameters.items() if k != 'password'}
        add_log("DEBUG", f"Snowpark Connection Parameters: {loggable_params}")

        session = Session.builder.configs(connection_parameters).create()
        add_log("SUCCESS", "Snowpark session created successfully.")
        st.sidebar.success("Snowpark session active!")
        return session
    except Exception as e:
        add_log("ERROR", f"Failed to create Snowpark session: {e}")
        st.sidebar.error(f"Could not connect to Snowflake. Please check connection details and password. Error: {e}")
        return None

# --- Get or Create Session ---
if st.session_state.snowpark_session is None:
    if SNOWFLAKE_ACCOUNT != "YOUR_SNOWFLAKE_ACCOUNT_IDENTIFIER" and SNOWFLAKE_USER != "YOUR_SNOWFLAKE_USER":
        st.session_state.snowpark_session = create_snowpark_session()
    else:
        st.warning("Snowflake connection details placeholders are not filled in. Cannot establish session.")
        add_log("WARN", "Skipping session creation due to placeholder values.")

session = st.session_state.snowpark_session if st.session_state.snowpark_session else None

# --- Sidebar Setup ---
st.sidebar.title("Controls")
# Use default key generation for toggle, relying on label
st.session_state.debug_mode = st.sidebar.toggle("Enable Debug Mode", value=st.session_state.get("debug_mode", False))
# Add explicit display of the state for debugging the toggle itself
st.sidebar.write(f"(Debug Mode State: {st.session_state.debug_mode})")

if st.sidebar.button("Clear Chat History & Log"):
    st.session_state.messages = []
    st.session_state.debug_log = []
    st.session_state.charts = []  # Clear saved visualizations
    st.session_state.visualization_states = {}  # Clear visualization states
    st.rerun()

# --- Application Title and Header ---
title_col1, title_col2 = st.columns([0.07, 0.93])
with title_col1:
    st.image("Q Logo 2024.png", width=45)
with title_col2:
    st.markdown("<h1 style='margin-top: -10px; margin-left: -15px;'>Chat with Cortex Agent</h1>", unsafe_allow_html=True)

# --- Helper Functions ---
def filter_messages_for_api(messages):
    """
    Creates a deep copy of messages, including assistant messages,
    filtering out only display-specific elements like fetched_table.
    """
    messages_copy = []
    types_to_remove = ["fetched_table"] # Only remove table markdown added client-side

    for msg in messages:
        msg_copy = copy.deepcopy(msg)
        if msg_copy["role"] == "assistant":
            # Filter specific content part types from assistant messages if needed
            if isinstance(msg_copy.get("content"), list):
                 msg_copy["content"] = [
                     part for part in msg_copy["content"]
                     if part.get("type", "").lower() not in types_to_remove
                 ]
        messages_copy.append(msg_copy)

    add_log("TRACE", f"Messages prepared for API: {messages_copy}")
    return messages_copy

def stream_text_and_collect_parts(sse_response: requests.Response, full_text: list, non_text_collector: list) -> Generator[str, None, str]:
    """
    Parses SSE stream, yields text deltas for st.write_stream (including text
    found within analyst1 tool results), collects non-text parts into the
    provided list (side effect), and returns the full accumulated text via
    StopIteration.value.

    Args:
        sse_response: The streaming requests.Response object.
        full_text: A list to be used to store the complete unfiltered text.
        non_text_collector: A list to append non-text content parts to.

    Yields:
        str: Text chunks (deltas) from message.delta events or analyst tool results,
             attempting to exclude raw related query text.

    Returns:
        str: The fully accumulated *unfiltered* text content from the stream (via StopIteration.value).
    """
    # Clear the lists at the start of processing this stream
    full_text.clear()
    non_text_collector.clear()
    client = SSEClient(sse_response)
    full_text_accumulator = "" # Accumulate full text internally (including RQs)
    add_log("DEBUG", "SSEClient initialized. Starting event iteration...")

    try:
        for event in client.events():
            add_log("TRACE", f"Received SSE Event: type='{event.event}', data='{event.data}'")

            if event.event == "message.delta":
                try:
                    if not event.data or event.data.isspace():
                        add_log("TRACE", "Skipping empty message.delta data.")
                        continue

                    event_data = json.loads(event.data)
                    delta_content_list = event_data.get("delta", {}).get("content", [])
                    add_log("TRACE", f"Parsed message.delta data: {delta_content_list}")

                    for part in delta_content_list:
                        part_type = part.get("type")
                        if not part_type: continue

                        text_delta_to_yield = None
                        text_delta_for_accumulator = None # Store text for accumulator separately

                        if part_type == "text":
                            text_delta = part.get("text", "")
                            if text_delta:
                                text_delta_for_accumulator = text_delta # Always accumulate original text
                                # Attempt to filter related queries from *yielded* text
                                # NOTE: This regex on small deltas might not reliably catch the full pattern.
                                if re.search(RELATED_QUERIES_REGEX, text_delta, re.DOTALL | re.IGNORECASE):
                                    add_log("TRACE", f"Potential RQ text detected in delta, not yielding: '{text_delta}'")
                                    pass # Don't set text_delta_to_yield
                                else:
                                    text_delta_to_yield = text_delta # Yield if not RQ

                        elif part_type == "tool_results":
                            tool_results = part.get("tool_results", {})
                            tool_name = tool_results.get("name")
                            # Check for text within analyst1 tool results
                            if tool_name == "analyst1":
                                try:
                                    inner_content_list = tool_results.get('content', [])
                                    if inner_content_list and isinstance(inner_content_list, list):
                                        inner_item = inner_content_list[0]
                                        inner_json = None
                                        if isinstance(inner_item, dict): inner_json = inner_item.get('json', {})
                                        elif isinstance(inner_item, str):
                                            try: inner_json = json.loads(inner_item).get('json', {})
                                            except json.JSONDecodeError: pass
                                        if isinstance(inner_json, dict) and "text" in inner_json:
                                            analyst_text_delta = inner_json.get("text", "")
                                            if analyst_text_delta:
                                                text_delta_for_accumulator = analyst_text_delta # Accumulate original
                                                # Check if this analyst text is also related query text (unlikely but possible)
                                                if re.search(RELATED_QUERIES_REGEX, analyst_text_delta, re.DOTALL | re.IGNORECASE):
                                                     add_log("TRACE", f"Potential RQ text detected in analyst delta, not yielding: '{analyst_text_delta}'")
                                                     pass
                                                else:
                                                     text_delta_to_yield = analyst_text_delta # Yield if not RQ
                                                add_log("TRACE", f"Found text in analyst1 tool_results: '{analyst_text_delta}'")
                                except Exception as e:
                                    add_log("WARN", f"Could not extract text from analyst1 tool_results: {e}")

                            # Collect the entire tool_results part regardless of text extraction
                            non_text_collector.append(part)
                            add_log("DEBUG", f"Collected non-text part: {part_type} (name: {tool_name})")

                        elif part_type in ["tool_use", "chart", "table", "fetched_table"]:
                            # Collect other non-text parts
                            content_part_data = {k: v for k, v in part.items() if k != 'index'}
                            if part_type == "fetched_table": content_part_data.setdefault("toolResult", False)
                            non_text_collector.append(content_part_data)
                            add_log("DEBUG", f"Collected non-text part: {part_type}")
                        else:
                            add_log("WARN", f"Unhandled delta part type: {part_type} - {part}")

                        # Accumulate original text delta if any text was processed
                        if text_delta_for_accumulator:
                            full_text_accumulator += text_delta_for_accumulator

                        # Yield filtered text delta if found
                        if text_delta_to_yield:
                            add_log("TRACE", f"Yielding text delta: '{text_delta_to_yield}'")
                            yield text_delta_to_yield

                except json.JSONDecodeError as json_err:
                    add_log("ERROR", f"JSON Parse Error in message.delta data: {json_err}\nData: {event.data}")
                except Exception as e:
                    add_log("ERROR", f"Error processing message.delta part: {e}\nData: {event.data}")

            elif event.event == "done":
                add_log("DEBUG", f"Received 'done' event. Data: {event.data}")
                break # Stop processing after 'done'
            else:
                add_log("WARN", f"Unhandled SSE event type: '{event.event}'")

    except requests.exceptions.ChunkedEncodingError as chunk_err:
         add_log("WARN", f"ChunkedEncodingError during SSE stream processing (might happen on stream end): {chunk_err}")
    except Exception as e:
        add_log("ERROR", f"Error iterating through SSE events: {e}")
        yield f"\n\nError processing stream: {e}" # Yield error message if something goes wrong
    finally:
        # Ensure the response is closed
        sse_response.close()
        add_log("DEBUG", "SSE stream processing finished or errored. Response closed.")

    # Store the complete *unfiltered* text in the provided list
    full_text.append(full_text_accumulator)
    # Return the accumulated *unfiltered* full text (will be the value of StopIteration)
    return full_text_accumulator

def execute_sql(sql_query: str) -> tuple[str | None, pd.DataFrame | None]:
    """ Executes SQL (after cleaning), gets query ID via query_history and results DataFrame. """
    global session
    if not session:
        add_log("ERROR", "Snowpark session not available for execute_sql.")
        return None, None

    query_id, dataframe = None, None
    add_log("DEBUG", f"Original SQL received: ```sql\n{sql_query}\n```")
    cleaned_sql = sql_query.strip()
    if cleaned_sql.endswith(';'): 
        cleaned_sql = cleaned_sql[:-1].rstrip()
    comment_to_remove = "-- Generated by Cortex Analyst"
    if comment_to_remove in cleaned_sql: 
        cleaned_sql = cleaned_sql.replace(comment_to_remove, "").strip()
    
    add_log("INFO", f"Attempting to execute cleaned SQL...")
    add_log("DEBUG", f"Cleaned SQL:\n```sql\n{cleaned_sql}\n```")

    try:
        add_log("DEBUG", "Executing SQL within session.query_history context...")
        with session.query_history(True) as query_history:
            snowpark_df = session.sql(cleaned_sql)
            dataframe = snowpark_df.to_pandas()
        add_log("DEBUG", "Finished SQL execution within context.")

        if query_history.queries:
            query_id = query_history.queries[-1].query_id
            add_log("SUCCESS", f"SQL OK. Query ID from history: {query_id}")
        else:
            add_log("WARN", "Query executed but query_history was empty.")
            query_id = session.last_query_id
            add_log("WARN", f"Using fallback session.last_query_id: {query_id}")

        if dataframe is None or dataframe.empty: add_log("INFO", "Query returned no results.")
        return query_id, dataframe

    except Exception as e:
        error_message = f"Error executing SQL: {e}\nQuery attempted:\n```sql\n{cleaned_sql}\n```"
        add_log("ERROR", error_message)
        last_qid = session.last_query_id if session else None
        add_log("INFO", f"Attempting to get last query ID after error: {last_qid}")
        return last_qid, None

def get_sql_exec_user_message(query_id: str) -> dict:
    """ Constructs the user message for SQL results (without ID). """
    return {
        "role": "user",
        "content": [{"type": "tool_results", "tool_results": {
            "name": "sql_exec",
            "content": [{"type": "json", "json": { "query_id": query_id }}]
        }}],
    }

def call_agent_api(messages_to_send: list, call_label: str) -> requests.Response | None:
    """
    Calls agent API using requests and the Snowpark session token (as requested).
    Includes "stream": True in the request body.
    Returns the streaming requests.Response object on success (status 200),
    otherwise returns None after logging the error.
    """
    global session 
    api_path = "/api/v2/cortex/agent:run"
    api_url = f"{SNOWFLAKE_API_BASE_URL}{api_path}"
    add_log("INFO", f"Making {call_label} API call to: {api_url}")

    session_token = None
    if not session:
        add_log("ERROR", f"{call_label} API call failed: Snowpark session is not available.")
        return None
    try:
        session_token = session.conf.get("rest").token
        if not session_token:
             raise ValueError("Retrieved session token is empty.")
        add_log("DEBUG", f"{call_label} Retrieved session token successfully (token value not logged).")
    except Exception as e:
        add_log("ERROR", f"{call_label} API call failed: Could not retrieve token from Snowpark session: {e}")
        return None

    # Use the REVERTED filter_messages_for_api
    messages_for_api = filter_messages_for_api(messages_to_send)
    request_body = {
        "model": MODEL_NAME,
        "messages": messages_for_api, # Now includes assistant context
        "tools": AGENT_TOOLS,
        "tool_resources": AGENT_TOOL_RESOURCES,
        "experimental": AGENT_EXPERIMENTAL_PARAMS,
        "stream": True # Need this here so Analyst responds with a stream...
    }
    headers = {
        "Authorization": f'Snowflake Token="{session_token}"',
        "Content-Type": "application/json",
        "Accept": "*/*", # Accept any response type (including text/event-stream)
    }

    add_log("DEBUG", f"{call_label} Request Body: {json.dumps(request_body, indent=2)}")
    loggable_headers = {k: (v[:18] + '..."' if k == "Authorization" else v) for k, v in headers.items()}
    add_log("DEBUG", f"{call_label} Request Headers: {loggable_headers}")

    try:
        # Make the request with stream=True
        response = requests.post(
            api_url,
            headers=headers,
            json=request_body,
            timeout=API_TIMEOUT_SECONDS,
            stream=True # IMPORTANT: Keep the connection open for streaming
        )
        add_log("DEBUG", f"{call_label} Raw API Response Status: {response.status_code}")

        # Check status code BEFORE returning the response object
        if response.status_code == 200:
            add_log("INFO", f"{call_label} API call successful (200). Returning response object for streaming.")
            return response # Return the response object for the caller to handle streaming
        else:
            # Read the error body if not 200
            try:
                 error_text = response.text # Read the full error text
            except Exception as read_err:
                 error_text = f"(Could not read error response body: {read_err})"
            error_details = f"Status Code: {response.status_code}. Response: {error_text}"
            add_log("ERROR", f"{call_label} API request failed: {error_details}")
            response.close() # Close the connection if not returning the object
            return None # Indicate failure

    except requests.exceptions.Timeout:
        add_log("ERROR", f"Error during {call_label} API call: Request timed out after {API_TIMEOUT_SECONDS} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        add_log("ERROR", f"Error during {call_label} API call (requests exception): {e}")
        return None
    except Exception as e:
        add_log("ERROR", f"Unexpected error during {call_label} API call: {e}")
        return None

def create_best_chart(df: pd.DataFrame) -> alt.Chart | pdk.Deck:
    """
    Analyze the DataFrame and create the most appropriate chart based on its structure.
    """
    try:
        # Detect column types
        col_types = detect_column_types(df)
        date_cols = col_types['date_cols']
        numeric_cols = col_types['numeric_cols']
        text_cols = col_types['text_cols']
        lat_cols = col_types['lat_cols']
        lon_cols = col_types['lon_cols']
        
        # First check for geographic data (lat/long)
        if len(lat_cols) >= 1 and len(lon_cols) >= 1:
            # Find additional columns for color/size if available
            potential_color_cols = [col for col in text_cols if any(term in col.lower() for term in ['status', 'type', 'category', 'segment'])]
            potential_size_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['amount', 'value', 'revenue', 'count'])]
            
            map_config = {
                'lat_col': lat_cols[0],
                'lon_col': lon_cols[0],
                'color_col': potential_color_cols[0] if potential_color_cols else None,
                'size_col': potential_size_cols[0] if potential_size_cols else None
            }
            
            return create_chart10(df, map_config)
        
        # Special handling for count/aggregation data with categorical breakdowns
        count_keywords = ['count', 'total', 'number', 'qty', 'quantity']
        has_count_col = any(any(keyword in col.lower() for keyword in count_keywords) for col in numeric_cols)
        
        # Check if we have categorical columns that might represent dimensions
        dimension_keywords = ['type', 'category', 'reason', 'gender', 'status', 'segment']
        potential_dimension_cols = [col for col in text_cols if any(keyword in col.lower() for keyword in dimension_keywords)]
        
        chart = None
        
        # Prioritize stacked bar chart for count/categorical data with breakdowns
        if has_count_col and len(potential_dimension_cols) >= 2:
            # Find the count column
            count_col = next(col for col in numeric_cols if any(keyword in col.lower() for keyword in count_keywords))
            # Use the first two dimension columns for the chart
            primary_dim = potential_dimension_cols[0]
            secondary_dim = potential_dimension_cols[1]
            
            # Create a stacked bar chart
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(f"{primary_dim}:N", 
                       sort=alt.EncodingSortField(field=count_col, op="sum", order="descending"),
                       title=primary_dim.replace('_', ' ').title()),
                y=alt.Y(f"{count_col}:Q",
                       stack="zero",
                       title=count_col.replace('_', ' ').title()),
                color=alt.Color(f"{secondary_dim}:N",
                              title=secondary_dim.replace('_', ' ').title()),
                tooltip=[primary_dim, secondary_dim, count_col]
            ).properties(
                title=f"{primary_dim.replace('_', ' ').title()} by {secondary_dim.replace('_', ' ').title()}"
            )
        
        # If no special case matched, fall back to original logic
        if not chart:
            # Get suggested chart type
            chart_type = suggest_chart_type(df)
            
            if chart_type == 'chart1' and len(date_cols) == 1 and len(numeric_cols) == 1:
                chart = create_chart1(df, {
                    'date_col': date_cols[0],
                    'numeric_col': numeric_cols[0]
                })
                
            elif chart_type == 'chart2' and len(date_cols) == 1 and len(numeric_cols) >= 2:
                chart = create_chart2(df, {
                    'date_col': date_cols[0],
                    'num_col1': numeric_cols[0],
                    'num_col2': numeric_cols[1]
                })
                
            elif chart_type == 'chart3' and len(date_cols) == 1 and len(numeric_cols) >= 1 and len(text_cols) == 1:
                chart = create_chart3(df, {
                    'date_col': date_cols[0],
                    'numeric_col': numeric_cols[0],
                    'text_col': text_cols[0]
                })
                
            elif chart_type == 'chart4' and len(date_cols) == 1 and len(numeric_cols) >= 1 and len(text_cols) >= 2:
                chart = create_chart4(df, {
                    'date_col': date_cols[0],
                    'numeric_col': numeric_cols[0],
                    'text_cols': text_cols
                })
                
            elif len(numeric_cols) >= 2 and len(text_cols) >= 1:
                # Default to scatter plot if we have multiple numeric columns and a categorical column
                chart = create_chart5(df, {
                    'num_col1': numeric_cols[0],
                    'num_col2': numeric_cols[1],
                    'text_col': text_cols[0]
                })
                
            elif len(numeric_cols) >= 1 and len(text_cols) >= 1:
                # Default to bar chart with selectable text column
                chart = create_chart9(df, {
                    'numeric_col': numeric_cols[0],
                    'text_cols': text_cols
                })
                
            # If no specific chart type matches, create a basic bar chart
            elif len(numeric_cols) >= 1:
                # Create a simple bar chart using the first numeric column
                chart = alt.Chart(df).mark_bar().encode(
                    y=alt.Y(numeric_cols[0] + ':Q', title=numeric_cols[0]),
                    x=alt.X(df.columns[0] + ':N', sort='-y', title=df.columns[0])
                ).properties(
                    title='Data Overview'
                )

        # Apply styling to the chart if one was created
        if chart:
            chart = chart.properties(
                width='container',
                height=400  # Increased height for better visibility of stacked bars
            ).configure_axis(
                labelColor='hsla(221, 18%, 44%, 1)',
                titleColor='hsla(221, 18%, 44%, 1)',
                labelFontSize=12,
                titleFontSize=14
            ).configure_legend(
                labelColor='hsla(221, 18%, 44%, 1)',
                titleColor='hsla(221, 18%, 44%, 1)',
                labelFontSize=12,
                titleFontSize=14,
                symbolSize=100  # Larger legend symbols
            ).configure_title(
                fontSize=16,
                color='hsla(221, 18%, 44%, 1)'
            ).configure_range(
                category=['hsla(216, 81%, 50%, 1)', 'hsla(47, 100%, 46%, 1)', 
                         'hsla(162, 53%, 55%, 1)', 'hsla(351, 83%, 45%, 1)', 
                         'hsla(250, 88%, 65%, 1)', 'hsla(25, 100%, 56%, 1)',
                         'hsla(194, 100%, 72%, 1)', 'hsla(284, 100%, 68%, 1)']
            )
        
        return chart
        
    except Exception as e:
        add_log("ERROR", f"Error creating best chart: {e}")
        return None

def display_non_text_content(content_parts: list, full_text_context: str | None = None, message_key_prefix: str = "msg"):
    """
    Renders charts, tables, related queries etc. from a list of content parts.
    Skips analyst1 tool results, fetched_table, search results as they are handled elsewhere or via citations.
    Optionally uses full_text_context to render citations/RQs.
    Removes st.info wrappers.
    """
    add_log("DEBUG", f"Displaying {len(content_parts)} non-text parts (in display_non_text_content).")
    analyst_text = ""
    all_citations_in_message = [] # For citation rendering

    # Pass 1: Extract data needed for context (citations, Analyst Text)
    for part in content_parts:
        content_type = part.get("type", "").lower()
        if content_type == "tool_results":
            tool_results_data = part.get('tool_results', {})
            tool_name = tool_results_data.get('name', 'N/A')
            try:
                inner_content_list = tool_results_data.get('content', [])
                if inner_content_list and isinstance(inner_content_list, list) and len(inner_content_list) > 0:
                    inner_item = inner_content_list[0]
                    inner_json = None
                    if isinstance(inner_item, dict): 
                        inner_json = inner_item.get('json', {})
                    elif isinstance(inner_item, str):
                        try: inner_json = json.loads(inner_item).get('json', {})
                        except json.JSONDecodeError: pass

                    if isinstance(inner_json, dict):
                        # Extract Citations from search results (even if name is empty)
                        # This populates all_citations_in_message for Pass 4
                        if "searchResults" in inner_json:
                            raw_citations = inner_json["searchResults"]
                            if isinstance(raw_citations, list):
                                for res in raw_citations:
                                    try: 
                                        num = int(res.get("source_id", -1))
                                        url = res.get("url"); text = res.get("text", "")
                                    except (ValueError, TypeError): 
                                        continue
                                    if num != -1: 
                                        all_citations_in_message.append({"number": num, "text": text, "url": url})
                        # Extract Text from analyst results
                        if tool_name == "analyst1":
                             if "text" in inner_json: 
                                 analyst_text += inner_json["text"] + "\n"
            except Exception as e: add_log("ERROR", f"Error extracting data from tool_results: {e}")

    # Pass 2: Render Analyst Interpretation Text
    if analyst_text:
        st.markdown(analyst_text)

    # Pass 3: Render other non-analyst parts (Charts, Other Tool Results)
    for i, part in enumerate(content_parts): # Use enumerate for unique key generation
        content_type = part.get("type", "").lower()
        tool_results_data = part.get('tool_results', {}) # Get tool results data for name check
        tool_name = tool_results_data.get('name', 'N/A')

        # Skip parts already handled or not meant for direct display here
        # Skip analyst1 results, text, tool_use, fetched_table, search results, and chart data
        if (content_type in ["tool_use", "text", "fetched_table", "chart"] or 
            tool_name in ["analyst1", "search1", "data_to_chart"] or 
            part.get("tool_results", {}).get("name") in ["search1", "data_to_chart"]): # Check both ways
             # Also check if it *contains* searchResults even if name is missing
             is_search_result = False
             if content_type == "tool_results":
                 try:
                     inner_json = tool_results_data.get('content', [{}])[0].get('json', {})
                     if isinstance(inner_json, dict) and "searchResults" in inner_json:
                         is_search_result = True
                 except Exception: pass # Ignore errors in check
             if is_search_result:
                 continue # Skip search results display here
             continue # Skip all other filtered types

        elif content_type == "tool_results": # Display other tool results (NOT search, NOT analyst, NOT data_to_chart)
             if tool_name == "data_to_chart":
                 add_log("DEBUG", "Skipping display of data_to_chart tool result JSON.")
             else: # Fallback for truly unknown tool results
                 # We do not know what we got, so just skip it
                 add_log("WARN", f"Raw JSON for unhandled tool result: {tool_name}: {part}")
        else:
            # We do not know what we got, so just skip it 
            add_log("WARN", f"Unknown non-text content type: {content_type}: {part}")

    # Pass 4: Render Citations & Related Queries if context is available
    if full_text_context:
        # --- MODIFIED: Define extraction functions locally ---
        def extract_related_queries(text: str) -> list:
            queries = []
            if not text: 
                return queries
            try:
                # Use regex defined globally
                for match in re.finditer(RELATED_QUERIES_REGEX, text, re.DOTALL | re.IGNORECASE):
                    queries.append({"relatedQuery": match.group(1).strip(), "answer": match.group(2).strip()})
            except Exception as e: add_log("WARN", f"Error extracting RQs: {e}")
            return queries

        def extract_relevant_citations(text: str, all_citations: list) -> list:
            relevant_citations = []
            if not text or not all_citations: return relevant_citations
            try:
                # Use regex defined globally
                used_numbers = {int(num) for num in re.findall(CITATION_REGEX, text)}
                if used_numbers:
                    relevant_citations = [ cit for cit in all_citations if isinstance(cit.get("number"), int) and cit.get("number") in used_numbers ]
                    relevant_citations.sort(key=lambda x: x.get("number", 0))
            except Exception as e: add_log("WARN", f"Error extracting citations: {e}")
            return relevant_citations
        # --- End local function definitions ---

        related_queries = extract_related_queries(full_text_context)
        relevant_citations = extract_relevant_citations(full_text_context, all_citations_in_message)

        # --- MODIFIED: Display Related Queries using Expanders ---
        if related_queries:
             st.markdown("---") # Add separator
             st.subheader("Related Queries") # Add header
             for idx, rq in enumerate(related_queries):
                 with st.expander(rq['relatedQuery']): # Use query as title
                     st.markdown(rq['answer']) # Display answer inside

        if relevant_citations:
            st.markdown("---") # Add separator
            with st.container(border=True):
                st.write("Citations")
                for cit in relevant_citations:
                    url = cit.get('url')
                    with st.expander(f"Citation 【†{cit.get('number')}†】", expanded=False):
                        st.markdown(f"{cit.get('text', 'Source')}" + (f" ([link]({url}))" if url else ""))

    else:
         add_log("DEBUG", "Skipping Citation/RQ rendering as full text context was not provided.")

def display_chart(chart_data, message_index, is_new=False):
    """Helper function to display a chart with its controls."""
    chart_id = chart_data['id']
    state_key = f"msg{message_index}_{chart_id}"
    
    # For historical charts, use the stored state if available
    if not is_new and state_key in st.session_state.visualization_states:
        stored_state = st.session_state.visualization_states[state_key]
        
        # Display the chart based on its type
        if isinstance(stored_state.get('chart'), pdk.Deck):
            st.pydeck_chart(stored_state['chart'])
        elif stored_state.get('chart'):
            st.altair_chart(stored_state['chart'], use_container_width=True)
        else:
            st.info("Could not create a visualization for this data structure.")
        
        return
    
    # For new charts or if no stored state exists
    # Initialize session state for this chart if needed
    if f"chart_data_{chart_id}" not in st.session_state:
        st.session_state[f"chart_data_{chart_id}"] = {
            'chart': chart_data['chart'],
            'data': chart_data['data'].copy()
        }
    
    # Display the chart based on its type
    current_chart = st.session_state[f"chart_data_{chart_id}"]['chart']
    if isinstance(current_chart, pdk.Deck):
        st.pydeck_chart(current_chart)
    elif current_chart:
        st.altair_chart(current_chart, use_container_width=True)
    else:
        st.info("Could not create a visualization for this data structure.")
    
    # Store the current state if this is a new chart
    if is_new:
        st.session_state.visualization_states[state_key] = {
            'chart': current_chart
        }

def is_sample_request(query: str) -> bool:
    """Check if the query is asking for a sample or example of data."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in SAMPLE_KEYWORDS)

# --- Core Chat Logic ---

# Display chat messages
st.markdown("---")

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Initialize containers for different content types
        full_text = ""
        non_text_parts = []
        sql_part = None
        table_part = None
        chart_data = None
        interpretation_text = None
        response_content = []
        
        # Find associated chart data first
        for chart in st.session_state.charts:
            if chart.get('message_index') == i:
                chart_data = chart
                break

        # First pass: Find interpretation text and collect all parts
        for part in message.get("content", []):
            part_type = part.get("type")
            if part_type == "text":
                text_content = part.get("text", "")
                if "This is our interpretation of your question:" in text_content:
                    interpretation_text = text_content
                else:
                    response_content.append({"type": "text", "text": text_content})
            elif part_type == "fetched_table":
                table_part = part
            elif part_type == "tool_results" and part.get("tool_results", {}).get('name') == 'analyst1':
                try:
                    inner_content = part['tool_results'].get('content', [{}])[0]
                    inner_json = None
                    if isinstance(inner_content, dict): 
                        inner_json = inner_content.get('json', {})
                    elif isinstance(inner_content, str):
                        try: 
                            inner_json = json.loads(inner_content).get('json', {})
                        except json.JSONDecodeError: 
                            pass
                    if isinstance(inner_json, dict) and "sql" in inner_json:
                        sql_part = part
                    else:
                        non_text_parts.append(part)
                except Exception:
                    non_text_parts.append(part)
            else:
                non_text_parts.append(part)

        # Second pass: Display content
        if interpretation_text:
            # Create main container for all content
            with st.container():
                # Display interpretation text first
                st.markdown(interpretation_text)
                
                # Create nested container for all content under interpretation
                with st.container():
                    # For historical messages, display content without expanders
                    if i < len(st.session_state.messages) - 1:  # If this is not the latest message
                        # Display regular text content
                        for content in response_content:
                            st.markdown(content["text"])
                        
                        # Display table if present
                        if table_part:
                            # Check if this was a sample request
                            is_sample = False
                            if i < len(st.session_state.messages):
                                is_sample = is_sample_request(st.session_state.messages[i]["content"][0]["text"])
                            
                            if is_sample:
                                # For sample data, display table directly
                                st.markdown(table_part.get("tableMarkdown"))
                            else:
                                # For non-sample data, use expander
                                with st.expander("View result table", expanded=False):
                                    st.markdown(table_part.get("tableMarkdown"))
                        
                        # Display chart if present and not a sample request
                        if chart_data and not is_sample:
                            st.markdown("---")
                            viz_tab, data_tab = st.tabs([f"Visualization {chart_data['id']}", f"Data {chart_data['id']}"])
                            
                            with viz_tab:
                                display_chart(chart_data, i, is_new=False)
                            
                            with data_tab:
                                st.dataframe(chart_data['data'])
                        
                        # Display other non-text content
                        if non_text_parts:
                            display_non_text_content(non_text_parts, "\n".join([r["text"] for r in response_content]), message_key_prefix=f"msg_{i}")
                    
                    else:  # For the latest message, show all expanders
                        # Display SQL if present (this will be at the top)
                        if sql_part:
                            try:
                                sql_to_view = sql_part['tool_results']['content'][0]['json']['sql']
                                with st.expander("View SQL"):
                                    st.code(sql_to_view, language="sql")
                            except Exception as e:
                                add_log("WARN", f"Error displaying SQL from history: {e}")
                        
                        # Display regular text content
                        for content in response_content:
                            st.markdown(content["text"])
                        
                        # Display table if present
                        if table_part:
                            # Check if this was a sample request
                            is_sample = False
                            if i < len(st.session_state.messages):
                                is_sample = is_sample_request(st.session_state.messages[i]["content"][0]["text"])
                            
                            if is_sample:
                                # For sample data, display table directly
                                st.markdown(table_part.get("tableMarkdown"))
                            else:
                                # For non-sample data, use expander
                                with st.expander("View result table", expanded=False):
                                    st.markdown(table_part.get("tableMarkdown"))
                        
                        # Display chart if present and not a sample request
                        if chart_data and not is_sample:
                            st.markdown("---")
                            viz_tab, data_tab = st.tabs([f"Visualization {chart_data['id']}", f"Data {chart_data['id']}"])
                            
                            with viz_tab:
                                display_chart(chart_data, i, is_new=False)
                            
                            with data_tab:
                                st.dataframe(chart_data['data'])
                        
                        # Display other non-text content
                        if non_text_parts:
                            display_non_text_content(non_text_parts, "\n".join([r["text"] for r in response_content]), message_key_prefix=f"msg_{i}")
        else:
            # If no interpretation text, display everything sequentially
            for content in response_content:
                st.markdown(content["text"])
            
            if sql_part:
                try:
                    sql_to_view = sql_part['tool_results']['content'][0]['json']['sql']
                    with st.expander("View SQL"):
                        st.code(sql_to_view, language="sql")
                except Exception as e:
                    add_log("WARN", f"Error displaying SQL from history: {e}")
            
            if table_part:
                # Check if this was a sample request
                is_sample = False
                if i < len(st.session_state.messages):
                    is_sample = is_sample_request(st.session_state.messages[i]["content"][0]["text"])
                
                if is_sample:
                    # For sample data, display table directly
                    st.markdown(table_part.get("tableMarkdown"))
                else:
                    # For non-sample data, use expander
                    with st.expander("View result table", expanded=False):
                        st.markdown(table_part.get("tableMarkdown"))
            
            if chart_data:
                st.markdown("---")
                viz_tab, data_tab = st.tabs([f"Visualization {chart_data['id']}", f"Data {chart_data['id']}"])
                
                with viz_tab:
                    display_chart(chart_data, i, is_new=False)
                
                with data_tab:
                    st.dataframe(chart_data['data'])
            
            if non_text_parts:
                display_non_text_content(non_text_parts, "\n".join([r["text"] for r in response_content]), message_key_prefix=f"msg_{i}")

# --- Handle User Input ---
if prompt := st.chat_input("What can I help with?"):
    # Append user message immediately to history for display
    user_message = { "role": "user", "content": [{"type": "text", "text": prompt}] }
    st.session_state.messages.append(user_message)
    # Rerun to display the user message and trigger processing below
    st.rerun()

# --- Process Turn: API Calls and Tool Execution (Reverted to Single Spinner / Placeholders) ---
# Check if the last message is from the user and needs processing
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    add_log("INFO", "Processing turn started...") # Add log entry here

    # Ensure session is available before proceeding
    if not session:
        st.error("Cannot process request: Snowflake connection is not established.")
        error_msg = "Sorry, I cannot process your request because the connection to Snowflake failed. Please check the configuration and logs."
        # Avoid adding duplicate error messages
        if not st.session_state.messages or st.session_state.messages[-1].get("role") != "assistant" or error_msg not in str(st.session_state.messages[-1].get("content",[])):
            st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": error_msg}]})
            st.rerun()
    else:
        # Get the latest messages to send to the API
        messages_to_send = st.session_state.messages

        # --- Assistant Response Area ---
        with st.chat_message("assistant"):
            # Initialize variables for this turn outside the spinner
            non_text_parts_1 = []
            non_text_parts_2 = []
            full_text_1_lst = [] # List to capture full text from generator 1
            full_text_2_lst = [] # List to capture full text from generator 2
            full_text_1 = "" # Unfiltered full text from stream 1
            full_text_2 = "" # Unfiltered full text from stream 2
            final_history_content = [] # Build the content list for the final message object
            sql_to_execute = None
            sql_dataframe_result = None
            query_id = None
            assistant_message_for_history = None # Initialize history message
            unique_non_text_to_display = [] # Initialize list for non-text display items
            sql_exec_success = False # Initialize flag
            api_call_2_success = False # Initialize flag

            # Use st.spinner for status indication during processing
            with st.spinner("Thinking..."):
                # === First API Call ===
                api_response_1 = call_agent_api(messages_to_send, call_label="1st")

            if api_response_1:
                # Stream text to UI, collect non-text parts, capture full text
                with st.spinner("Streaming..."):
                    try:
                        # Write stream directly, capture yielded text (potentially filtered by generator)
                        # The generator also populates non_text_parts_1 and full_text_1_lst via side effect
                        yielded_text_1 = st.write_stream(stream_text_and_collect_parts(api_response_1, full_text_1_lst, non_text_parts_1))
                        # Retrieve the full unfiltered text from the list populated by the generator
                        full_text_1 = full_text_1_lst[0] if full_text_1_lst else yielded_text_1 # Fallback to yielded text
                        add_log("DEBUG", f"Stream 1 finished. Full text length: {len(full_text_1)}, Collected non-text parts: {len(non_text_parts_1)}")
                    except Exception as e:
                            add_log("ERROR", f"Error during st.write_stream or processing stream 1: {e}")
                            # Update placeholder with error message
                            st.error(f"Error displaying response stream: {e}")
                            final_history_content = [{"type": "text", "text": f"Error processing response: {e}"}]

                # --- Process collected parts from Call 1 ---
                if not final_history_content: # Only proceed if no error during streaming
                    filtered_text_1 = re.sub(RELATED_QUERIES_REGEX, "", full_text_1, flags=re.DOTALL | re.IGNORECASE).strip()

                    # Build message content 1 (used for API call 2 context and potentially final history)
                    message_content_1 = []
                    if filtered_text_1: # Use filtered text
                        message_content_1.append({"type": "text", "text": filtered_text_1})
                    message_content_1.extend(non_text_parts_1) # Keep non-text parts as they were

                    # Check for SQL Execution Path
                    sql_tool_results_part = None # The specific tool_results part with SQL
                    for part in non_text_parts_1:
                        if part.get("type") == "tool_results" and part.get("tool_results", {}).get('name') == 'analyst1':
                                try:
                                    inner_content = part['tool_results'].get('content', [{}])[0]
                                    inner_json = None
                                    if isinstance(inner_content, dict): 
                                        inner_json = inner_content.get('json', {})
                                    elif isinstance(inner_content, str):
                                        try: inner_json = json.loads(inner_content).get('json', {})
                                        except json.JSONDecodeError: pass
                                    if isinstance(inner_json, dict) and "sql" in inner_json:
                                        sql_to_execute = inner_json["sql"]
                                        sql_tool_results_part = part # Store the part containing the SQL
                                        add_log("INFO", "SQL found for execution.")
                                        break # Found SQL
                                except Exception as e: add_log("WARN", f"Error checking tool_results for SQL: {e}")

                    # --- Execute SQL and potentially Call 2 (still inside spinner) ---
                    if sql_to_execute:
                        # Display SQL expander if SQL was generated
                        if sql_to_execute:
                            with st.expander("View SQL"):
                                st.code(sql_to_execute, language="sql")

                        with st.spinner("Executing SQL..."):
                            query_id, sql_dataframe_result = execute_sql(sql_to_execute)

                        if query_id:
                            # Display SQL results table immediately after execution (if successful)
                            if sql_dataframe_result is not None and not sql_dataframe_result.empty:
                                # Generate unique IDs for this result
                                result_id = f"result_{len(st.session_state.messages)}"
                                chart_id = f"chart_{len(st.session_state.charts)}"
                                
                                # Check if this is a sample request
                                is_sample = is_sample_request(st.session_state.messages[-1]["content"][0]["text"])
                                
                                # Create chart only if not a sample request
                                chart = None
                                if not is_sample:
                                    chart = create_best_chart(sql_dataframe_result)
                                    if chart:
                                        # When creating a new chart, store the message index
                                        st.session_state.charts.append({
                                            'id': chart_id,
                                            'chart': chart,
                                            'data': sql_dataframe_result,
                                            'query': sql_to_execute,
                                            'result_id': result_id,  # Link to the result
                                            'message_index': len(st.session_state.messages) - 1  # Store the message index
                                        })
                                
                                # Initialize expander states if not exists
                                if result_id not in st.session_state.table_expander_states:
                                    st.session_state.table_expander_states[result_id] = True
                                if chart_id not in st.session_state.chart_expander_states:
                                    st.session_state.chart_expander_states[chart_id] = True
                                
                                # Display current results
                                if is_sample:
                                    # For sample requests, only show the data table
                                    st.dataframe(sql_dataframe_result)
                                    # Store the table for history
                                    table_markdown = sql_dataframe_result.to_markdown(index=False)
                                    final_history_content.append({"type": "fetched_table", "tableMarkdown": table_markdown, "toolResult": True})
                                else:
                                    # Create tabs for visualization and data
                                    viz_tab, data_tab = st.tabs(["Visualization", "Data"])
                                    
                                    with data_tab:
                                        st.dataframe(sql_dataframe_result)
                                    
                                    with viz_tab:
                                        # Calculate current message index
                                        current_msg_index = len(st.session_state.messages) - 1
                                        if chart:
                                            chart_data = {
                                                'id': chart_id,
                                                'chart': chart,
                                                'data': sql_dataframe_result,
                                                'query': sql_to_execute
                                            }
                                            display_chart(chart_data, current_msg_index, is_new=True)

                            sql_exec_success = True # Mark SQL success
                            add_log("INFO", f"SQL executed (Query ID: {query_id}). Making 2nd API call.")
                            first_assistant_message = {"role": "assistant", "content": message_content_1} # Send filtered context
                            tool_result_user_message = get_sql_exec_user_message(query_id)

                            # Add data visualization request if we have results
                            if sql_dataframe_result is not None and not sql_dataframe_result.empty:
                                # Convert DataFrame to JSON for the data_to_chart tool
                                df_json = sql_dataframe_result.to_json(orient='records')
                                data_viz_message = {
                                    "role": "user",
                                    "content": [{
                                        "type": "tool_results",
                                        "tool_results": {
                                            "name": "data_to_chart",
                                            "content": [{
                                                "type": "json",
                                                "json": {
                                                    "data": df_json,
                                                    "request": "Create the most insightful visualization for this data"
                                                }
                                            }]
                                        }
                                    }]
                                }
                                messages_for_second_call = messages_to_send + [first_assistant_message, tool_result_user_message, data_viz_message]
                            else:
                                messages_for_second_call = messages_to_send + [first_assistant_message, tool_result_user_message]

                            # === Second API Call ===
                            with st.spinner("Analyzing data.."):
                                api_response_2 = call_agent_api(messages_for_second_call, call_label="2nd")

                            if api_response_2:        
                                try:
                                    # Write the second stream directly, capture yielded text
                                    yielded_text_2 = st.write_stream(stream_text_and_collect_parts(api_response_2, full_text_2_lst, non_text_parts_2))
                                    # Retrieve full unfiltered text
                                    full_text_2 = full_text_2_lst[0] if full_text_2_lst else yielded_text_2
                                    add_log("DEBUG", f"Stream 2 finished. Full text length: {len(full_text_2)}, Collected non-text parts: {len(non_text_parts_2)}")
                                    api_call_2_success = True # Mark API 2 success
                                except Exception as e:
                                    add_log("ERROR", f"Error during st.write_stream or processing stream 2: {e}")
                                    # Update second placeholder with error
                                    st.error(f"Error displaying final response stream: {e}")
                                    # Append error to final_history_content which will be built later
                                    api_call_2_success = False # Ensure this is false on error
                                    final_history_content.append({"type": "text", "text": f"Error processing final response: {e}"})

                                # --- Construct final history content (SQL Path) ---
                                # This happens regardless of stream 2 success, but uses flags
                                # Rebuild history content based on success flags
                                if api_call_2_success:
                                    combined_final_content = []
                                    # Add Text 1 (FILTERED) if it exists
                                    if filtered_text_1: combined_final_content.append({"type": "text", "text": filtered_text_1})
                                    # Add Tool Use/Result from Call 1
                                    combined_final_content.extend([p for p in non_text_parts_1 if p.get("type") in ["tool_use", "tool_results"]])
                                    # Add Fetched Table
                                    if sql_dataframe_result is not None and not sql_dataframe_result.empty:
                                        combined_final_content.append({ "type": "fetched_table", "tableMarkdown": sql_dataframe_result.to_markdown(index=False), "toolResult": True })
                                    # Add Text 2 (FILTERED) if it exists
                                    if full_text_2:
                                        filtered_text_2 = re.sub(RELATED_QUERIES_REGEX, "", full_text_2, flags=re.DOTALL | re.IGNORECASE).strip()
                                        if filtered_text_2: # Only add if non-empty after filtering
                                            combined_final_content.append({"type": "text", "text": filtered_text_2})
                                    # Add other non-text parts from Call 2
                                    combined_final_content.extend([p for p in non_text_parts_2 if p.get("type") not in ["tool_use", "tool_results"]])

                                    final_history_content = combined_final_content # Set the final history content
                                # else: Error text already appended if stream 2 failed

                            else: # Second API call failed
                                # Add error directly to history content list (using message_content_1 which has filtered text 1)
                                error_text = "\n\nSorry, I could not process the results of the SQL query."
                                final_history_content = message_content_1 + [{"type": "text", "text": error_text}]
                                # Display error in the second placeholder
                                st.error(error_text.strip())


                        else: # SQL execution failed
                            # Add error directly to history content list (using message_content_1 which has filtered text 1)
                            error_text = "\n\nSorry, I encountered an error executing the required SQL."
                            final_history_content = message_content_1 + [{"type": "text", "text": error_text}]
                            # Display error in the first placeholder (since no second stream happened)
                            st.error(error_text.strip())
                    # End of SQL execution block
                    else: # No SQL to execute, single step path
                        final_history_content = message_content_1 # History is content from call 1 (with filtered text)

            else: # First API call failed
                # Error message already displayed in placeholder
                # final_history_content was already set
                pass

            # --- Spinner finishes here ---

            # --- Append the fully constructed message to history ---
            if final_history_content and any(final_history_content):
                 assistant_message_for_history = {"role": "assistant", "content": final_history_content}
                 # Check if the last message is identical to avoid duplicates from reruns/errors
                 # Compare content only, as role will be the same
                 if not st.session_state.messages or st.session_state.messages[-1].get("content") != assistant_message_for_history.get("content"):
                      st.session_state.messages.append(assistant_message_for_history)
                      add_log("DEBUG", f"Appended final constructed message to history.")
                 else:
                      add_log("DEBUG", "Skipping append to history, message content seems identical to last one.")
            else:
                 add_log("WARN", "No final message content generated, not appending to history.")


            # --- Display Final Non-Text Content (after spinner) ---
            # The streamed text is already visible in the placeholders above.
            # Now display the non-text elements directly using st commands.
            add_log("DEBUG", "Displaying final non-text content...")
            all_non_text_parts = non_text_parts_1 + non_text_parts_2
            seen_json_strings = set()
            unique_non_text_to_display = []
            for part in all_non_text_parts:
                try:
                    # Use separators=(',', ':') for compact representation, sort keys
                    json_string = json.dumps(part, sort_keys=True, separators=(',', ':'))
                    if json_string not in seen_json_strings:
                        seen_json_strings.add(json_string)
                        unique_non_text_to_display.append(part)
                except TypeError: # Handle unhashable types if json.dumps fails
                    # Basic check to avoid duplicates if adding anyway
                    is_duplicate = False
                    for existing_part in unique_non_text_to_display:
                         if part == existing_part: is_duplicate = True; break
                    if not is_duplicate: unique_non_text_to_display.append(part)

            # Combine *original* full text for context display for citations/RQs
            # Use the original full_text_1 and full_text_2 captured before filtering
            combined_full_text_context = (full_text_1 or "") + ("\n" + (full_text_2 or "") if sql_to_execute else "")
            # Display the unique non-text parts, providing text context
            # Exclude SQL tool result and fetched_table as they are handled explicitly elsewhere
            parts_for_display_func = [p for p in unique_non_text_to_display if not (p.get("type") == "tool_results" and p.get("tool_results", {}).get("name") == "analyst1") and p.get("type") != "fetched_table"]
            # Pass message index/key if needed for unique button keys
            # Using a simple counter for now within the display function call
            display_non_text_content(parts_for_display_func, combined_full_text_context, message_key_prefix=f"turn_{len(st.session_state.messages)}")

# --- Conditional Debug Display in Sidebar --- MOVED TO HERE ---
# Add check for debug_mode before accessing debug_log
if st.session_state.get("debug_mode", False):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Debug Log")
    with st.sidebar.container(height=250):
        log_container = st.empty()
        log_text = ""
        # Get logs safely, default to empty list
        current_logs = st.session_state.get("debug_log", [])
        for entry in reversed(current_logs): log_text += entry + "\n"
        # Use a unique key based on log length to force refresh
        log_container.text_area("Log Output", value=log_text, height=230, key=f"debug_log_display_{len(current_logs)}", disabled=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Message History (Debug)")
    with st.sidebar.expander("Show Message History JSON", expanded=False):
        st.json(st.session_state.messages)
