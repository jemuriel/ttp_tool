import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Probabilistic_model.Improved_Prob_Model import Probabilistic_Model

st.set_page_config(layout="wide")
st.title("📦 Forecast Disaggregation Tool")

# --- Upload Raw Forecast CSV ---
uploaded_forecast_file = st.file_uploader("📤 Upload Raw Forecast CSV", type=['csv'])
if uploaded_forecast_file is None:
    st.warning("Please upload a raw forecast CSV file to continue.")
    st.stop()

# --- Load Raw Forecast ---
uploaded_raw_forecast = pd.read_csv(uploaded_forecast_file)
raw_forecast = uploaded_raw_forecast.copy()
raw_forecast["DATE"] = pd.to_datetime(raw_forecast["DATE"], format="%d/%m/%Y", errors="coerce")
raw_forecast["Month"] = raw_forecast["DATE"].dt.strftime("%Y-%m")

# --- Filters ---
month_options = sorted(raw_forecast["Month"].dropna().unique())
od_pair_options = sorted(raw_forecast["OD_PAIR"].dropna().unique())
preselected_od_pairs = ['AFT-BFT', 'BFT-AFT', 'MFT-AFT', 'MFT-PFT', 'MFT-BFT', 'MFT-SFT']
selected_months = st.multiselect("🕐 Select Month(s)", month_options, default=month_options)
selected_od_pairs = st.multiselect("🚛 Select OD Pairs", od_pair_options, default=preselected_od_pairs)

# --- Display Raw Forecast Table ---
filtered_raw = raw_forecast[
    raw_forecast["Month"].isin(selected_months) &
    raw_forecast["OD_PAIR"].isin(selected_od_pairs)
]

st.subheader("📋 Raw Forecast Data")
st.dataframe(filtered_raw, use_container_width=True)

# --- Raw Forecast Chart ---
st.subheader("📊 Raw Forecast Chart")
fig_original = px.line(
    filtered_raw,
    x="DATE", y="FORECASTED_TEUS", color="OD_PAIR", markers=True
)
fig_original.update_layout(xaxis_title="Month", yaxis_title="TEUs", xaxis_tickformat="%b %Y")
st.plotly_chart(fig_original, use_container_width=True)

# --- Run Disaggregation Model ---
@st.cache_data(show_spinner="Loading historical data...")
def get_historical_data():
    return pd.read_csv('csv_files/historical_data.csv')

@st.cache_data(show_spinner=False)
def run_disaggregation_model(raw_forecast, uploaded_service_timetable):
    historical_df = pd.read_csv('csv_files/historical_5years.csv')
    box_types_df = pd.read_csv('csv_files/box_types.csv')

    # st.dataframe(uploaded_service_timetable)
    # st.dataframe(raw_forecast)

    model = Probabilistic_Model(
        service_timetable=uploaded_service_timetable,
        historical_file=historical_df,
        forecast_file=raw_forecast,
        box_types_file=box_types_df,
    )
    result_df = model.run_complete_model()
    # result_df["saved_at"] = datetime.now().isoformat()
    # st.dataframe(result_df)
    return result_df

st.subheader("▶️ Run Disaggregation Model")

# --- Upload Service Timetable ---
uploaded_service_timetable = st.file_uploader("📤 Upload Service Timetable CSV", type=['csv'])

if uploaded_service_timetable is None:
    st.warning("Please upload a Service Timetable CSV file to continue.")
    st.stop()

# --- Show Button after Upload ---
if st.button("Run Disaggregation Model"):
    # selected_version_label = f"Uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # st.info(f"Running disaggregation model for forecast version: {selected_version_label}")

    # --- Load Raw Forecast from Uploaded File ---
    service_timetable = pd.read_csv(uploaded_service_timetable)

    # --- Run Disaggregation Model ---
    result_df = run_disaggregation_model(uploaded_raw_forecast, service_timetable)

    st.success(f"✅ Disaggregated forecast generated")

    # --- Process Results ---
    result_df["DATE"] = pd.to_datetime(result_df["DATE"], errors="coerce")
    result_df = result_df.dropna(subset=["DATE"])
    result_df["WEEK"] = result_df["DATE"].dt.to_period("W").apply(lambda r: r.start_time)
    result_df_weekly = result_df.groupby(["WEEK", "OD_PAIR", "SERVICE_ID", "BOX_TYPE"],
                                         as_index=False)[
        ["NUM_BOXES", "CALCULATED_TEUS"]].sum()

    st.session_state["disag_result_df"] = result_df
    st.session_state["disag_result_weekly_df"] = result_df_weekly

    # --- Download Link ---
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button("📅 Download Disaggregated Forecast", csv_buffer.getvalue().encode(),
                       "Disaggregated_forecast.csv")

# --- Display toggle and charts if results exist ---
if "disag_result_df" in st.session_state and "disag_result_weekly_df" in st.session_state:
    st.sidebar.header("📊 Filters (Disaggregated)")

    aggregate_weekly = st.toggle("📅 Show Results by Week", value=False)
    df_display = st.session_state["disag_result_weekly_df"] \
        if aggregate_weekly \
        else st.session_state["disag_result_df"]
    time_col = "WEEK" if aggregate_weekly else "DATE"

    reference_df = df_display[['OD_PAIR', 'SERVICE_ID', 'BOX_TYPE']].drop_duplicates()

    od_opts = sorted(reference_df["OD_PAIR"].dropna().unique())
    default_od_sel = [od for od in ['AFT-BFT', 'BFT-AFT', 'MFT-BFT'] if od in od_opts]
    selected_od_opts = st.sidebar.multiselect("OD Pair", od_opts, default=default_od_sel)

    filtered_services_df = reference_df[reference_df["OD_PAIR"].isin(selected_od_opts)]
    service_opts = sorted(filtered_services_df["SERVICE_ID"].dropna().unique())
    service_sel = st.sidebar.multiselect("Service ID", service_opts, default=service_opts)

    filtered_box_type_df = reference_df[reference_df["SERVICE_ID"].isin(service_sel)]
    box_opts = sorted(filtered_box_type_df["BOX_TYPE"].dropna().unique())
    box_sel = st.sidebar.multiselect("Box Type", box_opts, default=box_opts)

    dr = st.sidebar.date_input("Date Range", [df_display[time_col].min(), df_display[time_col].max()])

    df_filtered = df_display[
        (df_display[time_col] >= pd.to_datetime(dr[0])) &
        (df_display[time_col] <= pd.to_datetime(dr[1])) &
        (df_display["OD_PAIR"].isin(selected_od_opts)) &
        (df_display["SERVICE_ID"].isin(service_sel)) &
        (df_display["BOX_TYPE"].isin(box_sel))
    ]

    st.dataframe(df_filtered[[time_col, "SERVICE_ID", "BOX_TYPE", "OD_PAIR",
                              "CALCULATED_TEUS", "NUM_BOXES"]],
                 use_container_width=True,
                 key=f"df_forecast_{'_'.join(sorted(set(df_filtered['OD_PAIR'])))}"
                 )

    st.subheader("📦 Boxes by OD Pair")
    st.plotly_chart(
        px.line(
            df_filtered.groupby([time_col, "OD_PAIR"], as_index=False)["NUM_BOXES"].sum(),
            x=time_col, y="NUM_BOXES", color="OD_PAIR", markers=True
        ),
        use_container_width=True,
        key=f"chart_od_pair_{'_'.join(sorted(set(df_filtered['OD_PAIR'])))}"
    )

    st.subheader("🚆 Boxes by Service")
    st.plotly_chart(px.line(
        df_filtered.groupby([time_col, "SERVICE_ID"], as_index=False)["NUM_BOXES"].sum(),
        x=time_col, y="NUM_BOXES", color="SERVICE_ID", markers=True
        ),
        use_container_width = True,
        key = f"chart_service_{'_'.join(sorted(set(df_filtered['OD_PAIR'])))}"
    )

    st.subheader("📦 Boxes by Box Type")
    st.plotly_chart(px.line(
        df_filtered.groupby([time_col, "BOX_TYPE"], as_index=False)["NUM_BOXES"].sum(),
        x=time_col, y="NUM_BOXES", color="BOX_TYPE", markers=True
    ),
        use_container_width=True,
        key=f"chart_box_type_{'_'.join(sorted(set(df_filtered['OD_PAIR'])))}"
    )

else:
    st.info("Run the disaggregation model to see results.")
