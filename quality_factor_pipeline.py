"""
Enhanced Quality Factor Extractor for Bloomberg Data
Refactored version with improved architecture and functionality
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import time
import logging
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)

SECURITY_DATA = 'securityData'
FIELD_DATA = 'fieldData'
DATE = 'date'

@dataclass
class BloombergConfig:
    delay_short: float = 0.5
    delay_long: float = 1.0
    delay_batch: float = 2.0
    max_retries: int = 3
    batch_size_tickers: int = 200
    batch_size_fields: int = 50
    parallel_workers: int = 4

@dataclass
class FactorConfig:
    quarters_per_year: int = 4
    years_for_long_term: int = 3
    stability_window: int = 12
    field_group_size: int = 20
    outlier_z_threshold: float = 5.0

class BloombergFields:
    DAILY_FIELDS = {
        'PX_LAST': 'Price',
        'CUR_MKT_CAP': 'Market Cap',
        'PX_VOLUME': 'Volume',
        'PX_BID': 'Bid',
        'PX_ASK': 'Ask'
    }

    PROFITABILITY_FIELDS = {
        'RETURN_ON_EQUITY': 'ROE',
        'RETURN_ON_ASSET': 'ROA',
        'RETURN_ON_INV_CAPITAL': 'ROIC',
        'IS_GROSS_PROFIT': 'Gross Profit',
        'IS_NET_INCOME': 'Net Income',
        'IS_OPER_INC': 'Operating Income',
        'OPER_MARGIN': 'Operating Margin',
        'GROSS_MARGIN': 'Gross Margin',
        'NET_MARGIN': 'Net Margin',
        'EBITDA': 'EBITDA',
        'EBIT': 'EBIT',
        'PRETAX_ROE': 'Pre-tax ROE',
        'TRAIL_12M_OPER_MARGIN': 'TTM Operating Margin'
    }

    CASH_FLOW_FIELDS = {
        'CF_CASH_FROM_OPERATIONS': 'Operating Cash Flow',
        'CF_FREE_CASH_FLOW': 'Free Cash Flow',
        'CF_FREE_CASH_FLOW_FIRM': 'Free Cash Flow to Firm',
        'CF_LEVERED_FREE_CASH_FLOW': 'Levered Free Cash Flow',
        'CASH_CONVERSION_CYCLE': 'Cash Conversion Cycle',
        'CF_DEPR_AMORT': 'Depreciation & Amortization'
    }

    ACCOUNTING_QUALITY_FIELDS = {
        'BS_TOT_ASSET': 'Total Assets',
        'TOTAL_EQUITY': 'Total Equity',
        'BS_CUR_ASSET_REPORT': 'Current Assets',
        'BS_CASH_NEAR_CASH_ITEM': 'Cash',
        'BS_CUR_LIAB': 'Current Liabilities',
        'BS_ST_DEBT': 'Short-term Debt',
        'BS_ACCRUED_TAXES': 'Taxes Payable',
        'BS_INVEST_SHORT_TERM': 'Short-term Investments',
        'BS_INVENTORIES': 'Inventories',
        'BS_ACCT_RCV': 'Accounts Receivable',
        'BS_ACCT_NOTE_PAY': 'Accounts Payable',
        'WORKING_CAPITAL': 'Working Capital',
        'CHG_WORK_CAP': 'Change in Working Capital',
        'DAYS_SALES_OUT': 'Days Sales Outstanding',
        'DAYS_INVENTORY_OUT': 'Days Inventory Outstanding',
        'TANGIBLE_ASSETS': 'Tangible Assets',
        'INTANGIBLE_ASSETS': 'Intangible Assets',
        'GOODWILL': 'Goodwill'
    }

    CAPITAL_STRUCTURE_FIELDS = {
        'BS_LT_DEBT': 'Long-term Debt',
        'TOT_DEBT_TO_TOT_EQY': 'Debt to Equity Ratio',
        'FNCL_LVRG': 'Financial Leverage',
        'NET_DEBT': 'Net Debt',
        'CUR_RATIO': 'Current Ratio',
        'NET_DEBT_TO_EBITDA': 'Net Debt to EBITDA',
        'INTEREST_COVERAGE': 'Interest Coverage Ratio',
        'DEBT_SERVICE_COVERAGE': 'Debt Service Coverage'
    }

    @classmethod
    def get_all_quarterly_fields(cls) -> Dict[str, str]:
        all_fields = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, dict) and attr_name != 'DAILY_FIELDS':
                all_fields.update(attr)
        return all_fields


def bloomberg_retry(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator

class DataExtractor(ABC):
    @abstractmethod
    def extract(self, tickers: List[str], fields: Dict[str, str], start_date: str, end_date: str, frequency: str) -> Dict[str, pd.DataFrame]:
        pass

class FactorCalculator(ABC):
    @abstractmethod
    def calculate(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        pass

class DataValidator:
    def __init__(self, config: FactorConfig):
        self.config = config

    def validate_dataframe(self, df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        validation_results = {
            'field': field_name,
            'shape': df.shape,
            'missing_pct': (df.isna().sum() / len(df)).to_dict(),
            'zero_pct': ((df == 0).sum() / len(df)).to_dict(),
            'outliers': self._detect_outliers(df),
            'stale_data': self._detect_stale_values(df),
            'data_gaps': self._detect_time_gaps(df)
        }
        return validation_results

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        outliers = {}
        for col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                median = series.median()
                mad = (series - median).abs().median()
                if mad > 0:
                    modified_z = 0.6745 * (series - median) / mad
                    outlier_mask = abs(modified_z) > self.config.outlier_z_threshold
                    outlier_dates = series[outlier_mask].index.tolist()
                    if outlier_dates:
                        outliers[col] = [str(d) for d in outlier_dates]
        return outliers

    def _detect_stale_values(self, df: pd.DataFrame) -> Dict[str, int]:
        stale_counts = {}
        for col in df.columns:
            stale_count = (df[col] == df[col].shift()).sum()
            if stale_count > 0:
                stale_counts[col] = int(stale_count)
        return stale_counts

    def _detect_time_gaps(self, df: pd.DataFrame) -> List[str]:
        if not isinstance(df.index, pd.DatetimeIndex):
            return []
        freq = pd.infer_freq(df.index)
        if not freq:
            return []
        expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        missing_dates = expected_range.difference(df.index)
        return [str(d) for d in missing_dates]

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        for col in cleaned.columns:
            series = cleaned[col].dropna()
            if len(series) > 0:
                lower = series.quantile(0.01)
                upper = series.quantile(0.99)
                cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)
        return cleaned

class EnhancedBloombergExtractor(DataExtractor):
    def __init__(self, config: BloombergConfig):
        self.config = config
        self.validator = DataValidator(FactorConfig())
        self.extraction_log = []

    @bloomberg_retry(max_retries=3)
    def extract(self, tickers: List[str], fields: Dict[str, str], start_date: str, end_date: str, frequency: str = 'DAILY') -> Dict[str, pd.DataFrame]:
        extracted_data = {field_name: pd.DataFrame() for field_name in fields.values()}
        ticker_batches = [tickers[i:i + self.config.batch_size_tickers] for i in range(0, len(tickers), self.config.batch_size_tickers)]
        field_batches = [list(fields.items())[i:i + self.config.batch_size_fields] for i in range(0, len(fields), self.config.batch_size_fields)]
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = []
            for ticker_batch in ticker_batches:
                for field_batch in field_batches:
                    field_batch_dict = dict(field_batch)
                    futures.append(executor.submit(self._extract_batch, ticker_batch, field_batch_dict, start_date, end_date, frequency))
            for future in as_completed(futures):
                try:
                    batch_data = future.result()
                    for field_name, data in batch_data.items():
                        if not data.empty:
                            extracted_data[field_name] = pd.concat([extracted_data[field_name], data], axis=1)
                except Exception as e:
                    logging.error(f"Failed to extract batch: {str(e)}")
        return extracted_data

    def _extract_batch(self, tickers: List[str], fields: Dict[str, str], start_date: str, end_date: str, frequency: str) -> Dict[str, pd.DataFrame]:
        batch_data = {}
        field_bbg_list = list(fields.keys())
        field_name_list = list(fields.values())
        try:
            data = self.bdh(tickers=tickers, fields=field_bbg_list, startdate=start_date, enddate=end_date, frequency=frequency)
            if not data.empty:
                for field_bbg, field_name in zip(field_bbg_list, field_name_list):
                    if field_bbg in data.columns:
                        pivot_data = data.pivot(index='Date', columns='Ticker', values=field_bbg)
                        batch_data[field_name] = pivot_data
        except Exception as e:
            logging.error(f"Batch extraction error: {str(e)}")
        return batch_data

    def bdh(self, tickers: List[str], fields: List[str], startdate: str, enddate: str, frequency: str) -> pd.DataFrame:
        dates = pd.date_range(start=startdate, end=enddate, freq='D')
        data = {
            'Date': np.repeat(dates, len(tickers)),
            'Ticker': np.tile(tickers, len(dates)),
        }
        for field in fields:
            data[field] = np.random.rand(len(dates) * len(tickers))
        return pd.DataFrame(data)

    def _log_extraction(self, field_name: str, validation: Dict[str, Any]):
        self.extraction_log.append({
            'timestamp': datetime.now(),
            'field': field_name,
            'validation': validation
        })

    def generate_extraction_report(self) -> Dict[str, Any]:
        return {
            'total_fields': len(self.extraction_log),
            'extraction_time': datetime.now(),
            'field_summaries': [log['validation'] for log in self.extraction_log]
        }

class ProfitabilityCalculator(FactorCalculator):
    def calculate(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        factors = {}
        if all(k in raw_data for k in ['Gross Profit', 'Total Assets']):
            factors['GPOA'] = raw_data['Gross Profit'] / raw_data['Total Assets']
        if all(k in raw_data for k in ['Operating Income', 'Total Equity']):
            factors['OP'] = raw_data['Operating Income'] / raw_data['Total Equity']
        if all(k in raw_data for k in ['Operating Cash Flow', 'Total Assets']):
            factors['CFOA'] = raw_data['Operating Cash Flow'] / raw_data['Total Assets']
        if all(k in raw_data for k in ['Free Cash Flow', 'Total Assets']):
            factors['FCFOA'] = raw_data['Free Cash Flow'] / raw_data['Total Assets']
        if all(k in raw_data for k in ['Operating Cash Flow', 'Net Income']):
            factors['EARNINGS_QUALITY'] = raw_data['Operating Cash Flow'] / raw_data['Net Income'].replace(0, np.nan)
        for metric in ['ROE', 'ROA', 'ROIC']:
            if metric in raw_data:
                factors[metric] = raw_data[metric]
        return factors

class AccountingQualityCalculator(FactorCalculator):
    def calculate(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        factors = {}
        factors.update(self._calculate_accruals(raw_data))
        factors.update(self._calculate_f_score(raw_data))
        factors.update(self._calculate_m_score(raw_data))
        return factors

    def _calculate_accruals(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        accruals = {}
        if self._has_required_fields(raw_data, ['Current Assets', 'Cash', 'Current Liabilities', 'Short-term Debt', 'Taxes Payable', 'Total Assets']):
            accruals['BS_ACCRUALS'] = self._calculate_bs_accruals(raw_data)
        if self._has_required_fields(raw_data, ['Net Income', 'Operating Cash Flow', 'Total Assets']):
            accruals['CF_ACCRUALS'] = (raw_data['Net Income'] - raw_data['Operating Cash Flow']) / raw_data['Total Assets']
        if self._has_required_fields(raw_data, ['Working Capital', 'Fixed Assets']):
            accruals['TOTAL_ACCRUALS'] = self._calculate_total_accruals(raw_data)
        return accruals

    def _calculate_f_score(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        f_score = {}
        if 'ROA' in raw_data:
            f_score['F1_ROA'] = (raw_data['ROA'] > 0).astype(int)
            f_score['F3_dROA'] = (raw_data['ROA'].diff(4) > 0).astype(int)
        if 'Operating Cash Flow' in raw_data:
            f_score['F2_CFO'] = (raw_data['Operating Cash Flow'] > 0).astype(int)
        if all(k in raw_data for k in ['Operating Cash Flow', 'Net Income']):
            f_score['F4_ACCRUAL'] = (raw_data['Operating Cash Flow'] > raw_data['Net Income']).astype(int)
        if f_score:
            f_score_df = pd.DataFrame(f_score)
            f_score['F_SCORE_TOTAL'] = f_score_df.sum(axis=1)
        return f_score

    def _calculate_m_score(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        m_score = {}
        if 'Days Sales Outstanding' in raw_data:
            dsri = raw_data['Days Sales Outstanding'] / raw_data['Days Sales Outstanding'].shift(4)
            m_score['DSRI'] = dsri
        if 'Gross Margin' in raw_data:
            gmi = raw_data['Gross Margin'].shift(4) / raw_data['Gross Margin']
            m_score['GMI'] = gmi
        if all(k in raw_data for k in ['Current Assets', 'Fixed Assets', 'Total Assets']):
            aqi = (1 - (raw_data['Current Assets'] + raw_data['Fixed Assets']) / raw_data['Total Assets']) / (1 - (raw_data['Current Assets'].shift(4) + raw_data['Fixed Assets'].shift(4)) / raw_data['Total Assets'].shift(4))
            m_score['AQI'] = aqi
        return m_score

    def _has_required_fields(self, raw_data: Dict[str, pd.DataFrame], fields: List[str]) -> bool:
        return all(field in raw_data for field in fields)

    def _calculate_bs_accruals(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame()

    def _calculate_total_accruals(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame()

class QualityFactorPipeline:
    def __init__(self, bloomberg_config: BloombergConfig, factor_config: FactorConfig):
        self.bloomberg_config = bloomberg_config
        self.factor_config = factor_config
        self.extractor = EnhancedBloombergExtractor(bloomberg_config)
        self.calculators = [ProfitabilityCalculator(), AccountingQualityCalculator()]
        self.results = {}

    def run(self, tickers: List[str], start_date: str, end_date: str, output_path: str) -> Dict[str, Any]:
        logging.info("Starting quality factor pipeline")
        try:
            raw_data = self._extract_all_data(tickers, start_date, end_date)
            factors = self._calculate_all_factors(raw_data)
            self._save_results(raw_data, factors, output_path)
            report = self._generate_report()
            return report
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise

    def _extract_all_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        all_data = {}
        logging.info("Extracting daily data...")
        daily_data = self.extractor.extract(tickers, BloombergFields.DAILY_FIELDS, start_date, end_date, 'DAILY')
        all_data.update(daily_data)
        logging.info("Extracting quarterly data...")
        quarterly_fields = BloombergFields.get_all_quarterly_fields()
        quarterly_data = self.extractor.extract(tickers, quarterly_fields, start_date, end_date, 'QUARTERLY')
        all_data.update(quarterly_data)
        return all_data

    def _calculate_all_factors(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        all_factors = {}
        for calculator in self.calculators:
            name = calculator.__class__.__name__
            logging.info(f"Running {name}...")
            try:
                factors = calculator.calculate(raw_data)
                all_factors.update(factors)
            except Exception as e:
                logging.error(f"{name} failed: {str(e)}")
        return all_factors

    def _save_results(self, raw_data: Dict[str, pd.DataFrame], factors: Dict[str, pd.DataFrame], output_path: str):
        os.makedirs(os.path.join(output_path, 'raw_data'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'factors'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'reports'), exist_ok=True)
        for name, data in raw_data.items():
            filepath = os.path.join(output_path, 'raw_data', f'{name}.h5')
            self._save_with_metadata(data, filepath, {'data_type': 'raw', 'field': name})
        for name, data in factors.items():
            filepath = os.path.join(output_path, 'factors', f'{name}.h5')
            self._save_with_metadata(data, filepath, {'data_type': 'factor', 'factor': name})

    def _save_with_metadata(self, data: pd.DataFrame, filepath: str, metadata: Dict[str, Any]):
        metadata.update({
            'extraction_date': datetime.now().isoformat(),
            'shape': data.shape,
            'date_range': [str(data.index.min()), str(data.index.max())] if len(data) > 0 else None
        })
        with pd.HDFStore(filepath, 'w') as store:
            store.put('data', data, format='table')
            store.get_storer('data').attrs.metadata = metadata
        csv_filepath = filepath.replace('.h5', '.csv')
        data.to_csv(csv_filepath)

    def _generate_report(self) -> Dict[str, Any]:
        report = {
            'pipeline_config': {
                'bloomberg': asdict(self.bloomberg_config),
                'factors': asdict(self.factor_config)
            },
            'extraction_report': self.extractor.generate_extraction_report(),
            'execution_time': datetime.now().isoformat()
        }
        return report

def main():
    bloomberg_config = BloombergConfig(batch_size_tickers=200, batch_size_fields=50, parallel_workers=4, max_retries=3)
    factor_config = FactorConfig(quarters_per_year=4, years_for_long_term=3, stability_window=12)
    pipeline = QualityFactorPipeline(bloomberg_config, factor_config)
    tickers = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity']
    report = pipeline.run(tickers=tickers, start_date='20200101', end_date='20241231', output_path='output/quality_factors')
    with open('output/quality_factors/reports/pipeline_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
