import itertools
import logging
import os
import sys
from typing import List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Configure logging for debug and information messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Probabilistic_Model:
    def __init__(self, service_timetable, historical_file, forecast_file, box_types_file):
        """
        Initialize the model with input data and file paths.
        """
        self.service_timetable = service_timetable
        self.history_df = historical_file
        self.initial_forecast_df = forecast_file
        self.box_types = box_types_file
        # self.output_file = output_file

        # Attributes to be computed during processing
        self.aggregated_forecast = None
        self.history_df_teus = None
        self.service_names = None
        self.time_table_forecast = None
        self.container_probabilities = None
        self.teu_stats = None
        self.disaggregated_forecast = None

    def _validate_input_data(self):
        """
        Validate that all necessary columns are present in the input datasets.
        """
        required_forecast_cols = {'DATE', 'OD_PAIR', 'FORECASTED_TEUS'}
        required_history_cols = {'DATE', 'SERVICE_ID', 'ORIGIN', 'DESTINATION', 'BOX', 'NUM_TEUS'}
        required_timetable_cols = {'TRANSIT_LEG_ID', 'FREIGHT_ORIGIN_TERMINAL', 'FREIGHT_DESTINATION_TERMINAL'}
        required_box_cols = {'BOX', 'BOX_TYPE'}

        if not required_forecast_cols.issubset(self.initial_forecast_df.columns):
            raise ValueError(f"Forecast file missing columns: {required_forecast_cols - set(self.initial_forecast_df.columns)}")
        if not required_history_cols.issubset(self.history_df.columns):
            raise ValueError(f"History file missing columns: {required_history_cols - set(self.history_df.columns)}")
        if not required_timetable_cols.issubset(self.service_timetable.columns):
            raise ValueError(f"Timetable file missing columns: {required_timetable_cols - set(self.service_timetable.columns)}")
        if not required_box_cols.issubset(self.box_types.columns):
            raise ValueError(f"Box Types file missing columns: {required_box_cols - set(self.box_types.columns)}")

        logging.info("Input validation passed.")

    def _process_forecast_data(self):
        """
        Process and clean the forecast data.
        Converts dates and extracts corridors and month information.
        """
        logging.info("Processing forecast data...")
        self.initial_forecast_df['DATE'] = pd.to_datetime(self.initial_forecast_df['DATE'], dayfirst=True, errors='coerce')
        self.initial_forecast_df.dropna(subset=['DATE'], inplace=True)
        self.initial_forecast_df['MONTH'] = self.initial_forecast_df['DATE'].dt.month
        self.corridors = self.initial_forecast_df['OD_PAIR'].unique()
        logging.info(f"Forecast corridors: {len(self.corridors)} found.")

    def filter_historical_data(self):
        # FOR THE 5 YEARS FILE
        self.history_df = self.history_df.iloc[:, [0, 1, 2, 3, 12, 13]].copy()

        # Rename the columns
        self.history_df.columns = ['DESTINATION', 'DATE', 'SERVICE_ID', 'ORIGIN', 'BOX', 'NUM_TEUS']

    def _process_historical_data(self):
        """
        Process historical data by cleaning, merging box types, and computing subtotals.
        """
        logging.info("Processing historical data...")

        df = self.history_df.copy()
        df = df[['DESTINATION', 'DATE', 'SERVICE_ID', 'ORIGIN', 'BOX', 'NUM_TEUS']].dropna()
        df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
        df.dropna(subset=['DATE'], inplace=True)

        df['MONTH'] = df['DATE'].dt.month
        df = df.merge(self.box_types, on='BOX', how='left').dropna()
        df['OD_PAIR'] = df['ORIGIN'] + '-' + df['DESTINATION']
        df = df[df['OD_PAIR'].isin(self.corridors)]

        df = df.copy()
        df['BOX_TYPE'] = df['BOX_TYPE'].astype('Int64')
        df['BOX_TYPE'] = 'C' + df['BOX_TYPE'].astype(str)

        df.loc[:, 'WEEK'] = df['DATE'].dt.strftime('%U').astype(int)

        self.history_df_teus = df.copy()

        # Aggregate TEUs per service/date/week/box
        self.history_df = (
            df.groupby(['SERVICE_ID', 'DATE', 'MONTH', 'WEEK', 'BOX_TYPE', 'OD_PAIR'])
            .agg(SUBTOTAL_BOX=('NUM_TEUS', 'size'), SUBTOTAL_TEUS=('NUM_TEUS', 'sum'))
            .reset_index()
        )
        logging.info(f"Processed historical data shape: {self.history_df.shape}")

    def _aggregate_forecast_by_service(self):
        """
        Aggregate forecast by distributing corridor volumes into services using historical service shares.
        """
        logging.info("Aggregating forecast by service...")
        ref_service_corridor_dic = {
            row['TRANSIT_LEG_ID']: f"{row['FREIGHT_ORIGIN_TERMINAL']}-{row['FREIGHT_DESTINATION_TERMINAL']}"
            for _, row in self.service_timetable.iterrows()
        }
        # Correct OD_PAIR using service timetable mapping
        self.history_df['OD_PAIR'] = (self.history_df['SERVICE_ID'].map(ref_service_corridor_dic).
                                      fillna(self.history_df['OD_PAIR']))

        # Calculate service weights within each corridor
        service_weights = (
            self.history_df.groupby(['OD_PAIR', 'SERVICE_ID', 'MONTH'])['SUBTOTAL_TEUS']
            .sum().reset_index(name='TEUS_CORRIDOR_SERVICE')
        )
        corridor_totals = (
            self.history_df.groupby(['OD_PAIR', 'MONTH'])['SUBTOTAL_TEUS']
            .sum().reset_index(name='TEUS_CORRIDOR')
        )

        service_weights = service_weights.merge(corridor_totals, on=['OD_PAIR', 'MONTH'])
        service_weights['SERVICE_WEIGHT'] = service_weights['TEUS_CORRIDOR_SERVICE'] / service_weights['TEUS_CORRIDOR']

        # Apply weights to forecast
        merged_forecast = self.initial_forecast_df.merge(service_weights, on=['OD_PAIR', 'MONTH'], how='left')
        merged_forecast['NEW_FORECAST'] = merged_forecast['FORECASTED_TEUS'] * merged_forecast['SERVICE_WEIGHT']

        self.aggregated_forecast = (
            merged_forecast.groupby(['SERVICE_ID', 'MONTH', 'OD_PAIR'])
            .agg(WEIGHTED_FORECASTED_VALUE=('NEW_FORECAST', 'sum'), DATE=('DATE', 'first')).reset_index()
        )
        logging.info(f"Aggregated forecast shape: {self.aggregated_forecast.shape}")

    def _propagate_yearly_timetable(self):
        """
        Expand timetable for the entire year by mapping service IDs to their operational day of the week (DOW).
        Generates a full-year timetable forecast with DATE, DOW, SERVICE_ID, and OD_PAIR.
        """
        logging.info("Propagating timetable over the year...")

        # Generate OD_PAIR as 'Origin-Destination'
        self.service_timetable['OD_PAIR'] = (
                self.service_timetable['FREIGHT_ORIGIN_TERMINAL'].astype(str) + '-' +
                self.service_timetable['FREIGHT_DESTINATION_TERMINAL'].astype(str)
        )

        # Extract unique service IDs and map first character to Day of Week
        dow_mapping = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}
        self.service_timetable['DOW'] = self.service_timetable['TRANSIT_LEG_ID'].str[0].astype(int).map(dow_mapping)

        # Prepare the service reference table (DOW, SERVICE_ID, OD_PAIR)
        service_df = self.service_timetable[['DOW', 'TRANSIT_LEG_ID', 'OD_PAIR']].drop_duplicates()
        service_df.rename(columns={'TRANSIT_LEG_ID': 'SERVICE_ID'}, inplace=True)

        # Generate full date range for the year
        year = self.aggregated_forecast['DATE'].dt.year.iloc[0]
        date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31')

        # Create timetable with DATE, MONTH, WEEK, DOW
        timetable = pd.DataFrame({
            'DATE': date_range,
            'MONTH': date_range.month,
            'WEEK': date_range.strftime('%U').astype(int),
            'DOW': date_range.strftime('%a')
        })

        # Join timetable with services operating on the corresponding DOW
        self.time_table_forecast = timetable.merge(service_df, on='DOW', how='left').dropna(subset=['SERVICE_ID'])

        logging.info(f"Timetable forecast generated with {self.time_table_forecast.shape[0]} rows.")

    def _calc_probabilities(self):
        """
        Compute container presence probabilities per service and week.
        Fill missing weeks by copying rows from the nearest available week (forward/backward)
        replicating all BOX_TYPE rows.
        """
        logging.info("Calculating container probabilities...")

        # Compute counts per SERVICE_ID, OD_PAIR, WEEK, BOX_TYPE
        counts = (
            self.history_df[['DATE', 'SERVICE_ID', 'OD_PAIR', 'WEEK', 'BOX_TYPE']]
            .drop_duplicates()
            .groupby(['SERVICE_ID', 'OD_PAIR', 'WEEK', 'BOX_TYPE']).size().reset_index(name='COUNT')
        )

        # Total combinations per SERVICE_ID, OD_PAIR, WEEK
        total_combinations = (
            self.history_df[['DATE', 'SERVICE_ID', 'OD_PAIR', 'WEEK']]
            .drop_duplicates()
            .groupby(['SERVICE_ID', 'OD_PAIR', 'WEEK']).size().reset_index(name='TOTAL_COMBOS')
        )

        # Calculate probabilities
        container_prob = counts.merge(total_combinations, on=['SERVICE_ID', 'OD_PAIR', 'WEEK'])
        container_prob['PRESENT_PROB'] = container_prob['COUNT'] / container_prob['TOTAL_COMBOS']

        # Cartesian product: (SERVICE_ID, OD_PAIR) x WEEK (0-52)
        self.service_names = self.service_timetable[['TRANSIT_LEG_ID', 'OD_PAIR']].drop_duplicates()
        service_id_od_pairs = list(self.service_names.itertuples(index=False, name=None))
        complete_weeks = pd.DataFrame(
            itertools.product(service_id_od_pairs, range(0, 53)),
            columns=['SERVICE_OD', 'WEEK']
        )

        # Expand SERVICE_OD tuple into SERVICE_ID and OD_PAIR columns
        complete_weeks[['SERVICE_ID', 'OD_PAIR']] = pd.DataFrame(complete_weeks['SERVICE_OD'].tolist(),
                                                                 index=complete_weeks.index)
        complete_weeks = complete_weeks.drop(columns='SERVICE_OD')

        # Find which SERVICE_ID, OD_PAIR, WEEK combinations are missing
        existing_keys = container_prob[['SERVICE_ID', 'OD_PAIR', 'WEEK']].drop_duplicates()
        missing_keys = complete_weeks.merge(existing_keys, on=['SERVICE_ID', 'OD_PAIR', 'WEEK'], how='left',
                                            indicator=True)
        missing_keys = missing_keys[missing_keys['_merge'] == 'left_only'].drop(columns='_merge')

        # Fill missing weeks by replicating nearest week rows (forward/backward)
        filled_rows = []
        for _, row in missing_keys.iterrows():
            service_id, od_pair, target_week = row['SERVICE_ID'], row['OD_PAIR'], row['WEEK']

            # Subset with existing weeks for this service and OD_PAIR
            candidate_weeks = container_prob[
                (container_prob['SERVICE_ID'] == service_id) &
                (container_prob['OD_PAIR'] == od_pair)
                ]

            if candidate_weeks.empty:
                continue  # Cannot fill if no data for this service/od_pair at all

            # Find nearest week (by absolute distance)
            candidate_weeks = candidate_weeks.copy()
            candidate_weeks['WEEK_DIFF'] = (candidate_weeks['WEEK'] - target_week).abs()
            nearest_week = candidate_weeks.loc[candidate_weeks['WEEK_DIFF'].idxmin()]['WEEK']

            # Get all rows from the nearest week for this service/od_pair
            nearest_rows = container_prob[
                (container_prob['SERVICE_ID'] == service_id) &
                (container_prob['OD_PAIR'] == od_pair) &
                (container_prob['WEEK'] == nearest_week)
                ].copy()

            # Replace WEEK with the missing target week
            nearest_rows['WEEK'] = target_week
            nearest_rows['IS_ESTIMATED'] = True  # Mark as filled/estimated

            filled_rows.append(nearest_rows)

        # Combine original data and filled rows
        filled_rows_df = pd.concat(filled_rows, ignore_index=True) if filled_rows else pd.DataFrame(
            columns=container_prob.columns)
        container_prob['IS_ESTIMATED'] = False  # Original data mark as not estimated
        merged_container_prob = pd.concat([container_prob, filled_rows_df], ignore_index=True)

        # Ensure final output has no missing weeks
        merged_container_prob = merged_container_prob.sort_values(
            ['SERVICE_ID', 'OD_PAIR', 'WEEK', 'BOX_TYPE']).reset_index(drop=True)

        self.container_probabilities = merged_container_prob
        logging.info(f"Container probabilities shape: {self.container_probabilities.shape}")

    def _calc_avg_teu_weight(self, gamma=0.9):
        """
        Compute TEU weights by applying an exponential decay to historical data.
        Calculate week weights and intra-week box distributions.
        """
        logging.info("Calculating TEU weights with decay factor...")
        df = self.history_df.copy()
        df['YEAR'] = df['DATE'].dt.year
        df.sort_values(by=['SERVICE_ID', 'OD_PAIR', 'BOX_TYPE', 'YEAR'],
                       ascending=[True, True, True, False], inplace=True)
        df['YEAR_OFFSET'] = (df.groupby(['SERVICE_ID', 'OD_PAIR', 'BOX_TYPE'])['YEAR'].
                             rank(method='dense', ascending=False).astype(int) - 1)
        df['YEAR_WEIGHT'] = gamma ** df['YEAR_OFFSET']

        teu_stats = df.groupby(['SERVICE_ID', 'OD_PAIR', 'MONTH', 'WEEK', 'BOX_TYPE'], as_index=False).agg(
            TOTAL_TEUS_BOX=('SUBTOTAL_TEUS', 'sum'),
            YEAR_WEIGHT=('YEAR_WEIGHT', 'mean')
        )
        teu_stats['TOTAL_TEUS_BOX'] *= teu_stats['YEAR_WEIGHT']

        # Compute totals for normalization
        month_totals = (teu_stats.groupby(['SERVICE_ID', 'OD_PAIR', 'MONTH'])['TOTAL_TEUS_BOX'].
                        sum().reset_index(name='TOTAL_TEUS_MONTH'))
        week_totals = (teu_stats.groupby(['SERVICE_ID', 'OD_PAIR', 'MONTH', 'WEEK'])['TOTAL_TEUS_BOX'].
                       sum().reset_index(name='TOTAL_TEUS_WEEK'))

        teu_stats = teu_stats.merge(month_totals, on=['SERVICE_ID', 'OD_PAIR', 'MONTH'])
        teu_stats = teu_stats.merge(week_totals, on=['SERVICE_ID', 'OD_PAIR', 'MONTH', 'WEEK'])

        teu_stats['WEEK_WEIGHT'] = teu_stats['TOTAL_TEUS_WEEK'] / teu_stats['TOTAL_TEUS_MONTH']
        teu_stats['BOX_INTRA_WEEK_WEIGHT'] = teu_stats['TOTAL_TEUS_BOX'] / teu_stats['TOTAL_TEUS_WEEK']

        self.teu_stats = teu_stats
        logging.info(f"TEU stats shape: {self.teu_stats.shape}")

    def _probabilistic_rounding(self, x):
        """
        Apply probabilistic rounding to avoid systematic bias.
        """
        int_part = int(np.floor(x))
        decimal_part = x - int_part
        return int_part + (np.random.rand() < decimal_part)

    def _create_forecast(self):
        """
        Main disaggregation function: merges all components, extrapolates probabilities, applies TEU weights,
        performs reconciliation, and outputs the final forecast.
        """
        logging.info("Creating disaggregated forecast...")

        # Merge timetable with container probabilities
        disag_forecast = self.time_table_forecast.merge(
            self.container_probabilities[['SERVICE_ID', 'OD_PAIR', 'WEEK', 'BOX_TYPE', 'PRESENT_PROB']],
            on=['SERVICE_ID', 'WEEK', 'OD_PAIR'], how='left'
        )

        # --- Fill missing BOX_TYPEs by replicating nearest week rows ---
        missing_box_rows = disag_forecast[disag_forecast['BOX_TYPE'].isna()]
        filled_rows = []
        for _, row in missing_box_rows.iterrows():
            service_id, od_pair, target_week = row['SERVICE_ID'], row['OD_PAIR'], row['WEEK']

            candidate_weeks = self.container_probabilities[
                (self.container_probabilities['SERVICE_ID'] == service_id) &
                (self.container_probabilities['OD_PAIR'] == od_pair)
                ].copy()

            if candidate_weeks.empty:
                continue

            candidate_weeks['WEEK_DIFF'] = (candidate_weeks['WEEK'] - target_week).abs()
            nearest_week = candidate_weeks.loc[candidate_weeks['WEEK_DIFF'].idxmin()]['WEEK']

            nearest_rows = self.container_probabilities[
                (self.container_probabilities['SERVICE_ID'] == service_id) &
                (self.container_probabilities['OD_PAIR'] == od_pair) &
                (self.container_probabilities['WEEK'] == nearest_week)
                ].copy()

            for _, ref_row in nearest_rows.iterrows():
                filled_row = row.copy()
                filled_row['BOX_TYPE'] = ref_row['BOX_TYPE']
                filled_row['PRESENT_PROB'] = ref_row['PRESENT_PROB']
                filled_rows.append(filled_row)

        if filled_rows:
            filled_rows_df = pd.DataFrame(filled_rows)
            disag_forecast = disag_forecast[disag_forecast['BOX_TYPE'].notna()]
            disag_forecast = pd.concat([disag_forecast, filled_rows_df], ignore_index=True)

        # Merge forecast targets
        disag_forecast = disag_forecast.merge(
            self.aggregated_forecast[['SERVICE_ID', 'OD_PAIR', 'MONTH', 'WEIGHTED_FORECASTED_VALUE']],
            on=['SERVICE_ID', 'OD_PAIR', 'MONTH'], how='left'
        )

        # Merge TEU weights
        disag_forecast = disag_forecast.merge(
            self.teu_stats[
                ['SERVICE_ID', 'OD_PAIR', 'MONTH', 'WEEK', 'BOX_TYPE', 'WEEK_WEIGHT', 'BOX_INTRA_WEEK_WEIGHT']],
            on=['SERVICE_ID', 'OD_PAIR', 'MONTH', 'WEEK', 'BOX_TYPE'], how='left'
        )

        # Fill missing WEEK_WEIGHT and BOX_INTRA_WEEK_WEIGHT with fallback values
        disag_forecast['WEEK_WEIGHT'] = disag_forecast.groupby(['SERVICE_ID', 'OD_PAIR'])['WEEK_WEIGHT'] \
            .transform(lambda x: x.ffill().bfill())
        disag_forecast['WEEK_WEIGHT'] = disag_forecast['WEEK_WEIGHT'].fillna(
            1 / disag_forecast['WEEK_WEIGHT'].nunique())

        disag_forecast['BOX_INTRA_WEEK_WEIGHT'] = disag_forecast.groupby(['SERVICE_ID', 'OD_PAIR']) \
            ['BOX_INTRA_WEEK_WEIGHT'].transform(lambda x: x.ffill().bfill())
        disag_forecast['BOX_INTRA_WEEK_WEIGHT'] = disag_forecast['BOX_INTRA_WEEK_WEIGHT'] \
            .fillna(1 / disag_forecast['BOX_INTRA_WEEK_WEIGHT'].nunique())

        disag_forecast['WEIGHTED_FORECASTED_VALUE'] = disag_forecast.groupby(['SERVICE_ID', 'OD_PAIR']) \
            ['WEIGHTED_FORECASTED_VALUE'].transform(lambda x: x.ffill().bfill())
        disag_forecast['WEIGHTED_FORECASTED_VALUE'] = disag_forecast['WEIGHTED_FORECASTED_VALUE'] \
            .fillna(1 / disag_forecast['WEIGHTED_FORECASTED_VALUE'].nunique())

        disag_forecast['WEIGHTED_FORECASTED_VALUE'] = disag_forecast['WEIGHTED_FORECASTED_VALUE'].fillna(0)

        # Calculate initial CALCULATED_TEUS
        disag_forecast['CALCULATED_TEUS'] = (
                disag_forecast['WEEK_WEIGHT'] *
                disag_forecast['BOX_INTRA_WEEK_WEIGHT'] *
                disag_forecast['WEIGHTED_FORECASTED_VALUE']
        )

        # --- Reconcile CALCULATED_TEUS with self.initial_forecast quantities ---
        disag_teu_sum = disag_forecast.groupby(['OD_PAIR', 'MONTH'])['CALCULATED_TEUS'].sum().reset_index(
            name='DISAG_SUM')

        correction_factors = disag_teu_sum.merge(
            self.initial_forecast_df[['OD_PAIR', 'MONTH', 'FORECASTED_TEUS']],
            on=['OD_PAIR', 'MONTH'],
            how='left'
        )

        correction_factors['DISAG_SUM'] = correction_factors['DISAG_SUM'].replace(0, 1e-6)
        correction_factors['FACTOR'] = correction_factors['FORECASTED_TEUS'] / correction_factors['DISAG_SUM']
        correction_factors['FACTOR'] = correction_factors['FACTOR'].clip(lower=0, upper=100)

        disag_forecast = disag_forecast.merge(
            correction_factors[['OD_PAIR', 'MONTH', 'FACTOR']],
            on=['OD_PAIR', 'MONTH'],
            how='left'
        )

        disag_forecast['CALCULATED_TEUS'] = disag_forecast['CALCULATED_TEUS'] * disag_forecast['FACTOR']
        disag_forecast = disag_forecast.drop(columns=['FACTOR'])

        # Merge TEUs per box and ensure no NaNs
        mean_teus_per_box = self.history_df_teus.groupby(['BOX_TYPE'])['NUM_TEUS'].mean().reset_index(
            name='MEAN_TEUS_PER_BOX')
        disag_forecast = disag_forecast.merge(mean_teus_per_box, on='BOX_TYPE', how='left')
        global_teus_per_box = self.history_df_teus['NUM_TEUS'].mean()
        disag_forecast['MEAN_TEUS_PER_BOX'] = disag_forecast['MEAN_TEUS_PER_BOX'].fillna(global_teus_per_box)

        # Compute NUM_BOXES ensuring no NaNs
        disag_forecast['NUM_BOXES'] = disag_forecast['CALCULATED_TEUS'] / disag_forecast['MEAN_TEUS_PER_BOX']
        disag_forecast['NUM_BOXES'] = disag_forecast['NUM_BOXES'].fillna(0)

        # Apply probabilistic rounding
        disag_forecast['NUM_BOXES'] = disag_forecast['NUM_BOXES'].apply(self._probabilistic_rounding)

        # logging.info(f"Disaggregated forecast saved to {self.output_file}")

        self.disaggregated_forecast = disag_forecast[['DATE', 'MONTH', 'WEEK', 'DOW', 'SERVICE_ID', 'OD_PAIR',
                                                      'BOX_TYPE', 'CALCULATED_TEUS', 'MEAN_TEUS_PER_BOX', 'NUM_BOXES']]
        return self.disaggregated_forecast

    def _validate_disaggregated_forecast(self):
        """
        Validate that:
        1. Monthly OD totals match the original forecast.
        2. Every service runs across the full year per timetable.
        3. Services missing in history are flagged as 'estimated'.
        """
        logging.info("Validating disaggregated forecast...")

        # 1. Validate Monthly OD totals
        original_totals = self.initial_forecast_df.groupby(['OD_PAIR', 'MONTH'])['FORECASTED_TEUS'].sum().reset_index()
        disaggregated_totals = self.disaggregated_forecast.groupby(['OD_PAIR', 'MONTH'])[
            'CALCULATED_TEUS'].sum().reset_index()

        merged_totals = original_totals.merge(disaggregated_totals, on=['OD_PAIR', 'MONTH'], how = 'left')
        merged_totals['DIFFERENCE'] = merged_totals['FORECASTED_TEUS'] - merged_totals['CALCULATED_TEUS']

        if (merged_totals['DIFFERENCE'].abs() > 1).any():
            logging.error("Validation failed: Monthly OD totals do not match original forecast.")
            logging.error(merged_totals[merged_totals['DIFFERENCE'].abs() > 1])
        else:
            logging.info("Validation passed: Monthly OD totals match the original forecast.")

        # 2. Validate service full-year coverage
        expected_weeks = set(range(0, 53))
        services_missing_weeks = []

        # Calculate valid days for week 0 and week 52 based on start of year
        self.year = pd.to_datetime(self.time_table_forecast['DATE'].iloc[0]).year
        year_start_date = datetime(self.year, 1, 1)
        start_week_day = year_start_date.weekday()  # Monday=0, Sunday=6

        # Days present in Week 0
        week_0_days = set((start_week_day + d) % 7 for d in range(7 - start_week_day))

        # Days present in Week 52 (last week)
        end_of_year = datetime(self.year, 12, 31)
        end_week_day = end_of_year.weekday()
        week_52_days = set((end_week_day - d) % 7 for d in range(end_week_day + 1))

        # Iterate over services
        for service in self.service_names['TRANSIT_LEG_ID'].unique():
            service_weeks = set(
                self.disaggregated_forecast[self.disaggregated_forecast['SERVICE_ID'] == service]['WEEK'].unique()
            )

            # Get days of week this service runs (from service timetable)
            service_days = set(
                self.service_timetable[self.service_timetable['TRANSIT_LEG_ID'] == service]['DOW'].unique()
            )

            # Determine weeks expected for this service, considering boundary weeks
            service_expected_weeks = expected_weeks.copy()

            # Handle Week 0 exclusions
            if service_days.isdisjoint(week_0_days):
                service_expected_weeks.discard(0)

            # Handle Week 52 exclusions
            if service_days.isdisjoint(week_52_days):
                service_expected_weeks.discard(52)

            # Now validate missing weeks
            missing_weeks = service_expected_weeks - service_weeks
            if missing_weeks:
                services_missing_weeks.append((service, missing_weeks))

        # 3. Validate estimated services are flagged
        # estimated_services = self.disaggregated_forecast[self.disaggregated_forecast['SOURCE'] == 'estimated'][
        #     'SERVICE_ID'].unique()
        # services_without_history = set(self.service_names) - set(self.history_df['SERVICE_ID'].unique())
        #
        # if not set(estimated_services).issuperset(services_without_history):
        #     logging.error("Validation failed: Some services missing in history are not flagged as 'estimated'.")
        #     missing_flags = services_without_history - set(estimated_services)
        #     logging.error(f"Services not flagged as 'estimated': {missing_flags}")
        # else:
        #     logging.info("Validation passed: All missing services are flagged as 'estimated'.")

    def run_complete_model(self):
        """
        Execute the full model pipeline from validation to final disaggregation output.
        """
        self.filter_historical_data()
        self._validate_input_data()
        self._process_forecast_data()
        self._process_historical_data()
        self._aggregate_forecast_by_service()
        self._propagate_yearly_timetable()
        self._calc_probabilities()
        self._calc_avg_teu_weight()
        final_forecast = self._create_forecast()
        self._validate_disaggregated_forecast()

        # final_forecast.to_csv(r"C:\Users\61432\Downloads\forecast_dissag__.csv")
        return final_forecast

        # Save the final disaggregated forecast
        # final_forecast.to_csv(self.output_file, index=False)


# historical_df = pd.read_csv(r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning"\
#                             r"\DataFiles\Outbound_5years.csv")
#
# service_timetable = pd.read_csv(r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning"\
#                             r"\Opti_files\2. transit_leg_timetable.csv")
#
# box_types_df = pd.read_csv(r"C:\Users\61432\OneDrive - Pacific National" \
#                     r"\Tactical_Train_Planning\Access_to_Python\csv_files\box_types.csv")
# raw_forecast = pd.read_csv(r"C:\Users\61432\OneDrive - Pacific National\Tactical_Train_Planning"
#                            r"\Opti_files\1. raw_forecast_Tom.csv")
# output = r"C:\Users\61432\Downloads\forecast_dissag.csv"
#
# model = Probabilistic_Model(
#     service_timetable=service_timetable,
#     historical_file=historical_df,
#     forecast_file=raw_forecast,
#     box_types_file=box_types_df,
#     output_file=output
# )
# result_df = model.run_complete_model()



