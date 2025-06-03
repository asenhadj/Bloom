import pandas as pd
import os
from quality_factor_pipeline import QualityFactorPipeline, BloombergConfig, FactorConfig


def load_tickers_by_decade(csv_path: str):
    df = pd.read_csv(csv_path)
    decade_groups = df.groupby('Decade')['Ticker'].apply(list).to_dict()
    return decade_groups


def main():
    tickers_by_decade = load_tickers_by_decade('tickers.csv')

    bloomberg_config = BloombergConfig(
        batch_size_tickers=200,
        batch_size_fields=50,
        parallel_workers=4,
        max_retries=3
    )

    factor_config = FactorConfig(
        quarters_per_year=4,
        years_for_long_term=3,
        stability_window=12
    )

    pipeline = QualityFactorPipeline(bloomberg_config, factor_config)

    decade_dates = {
        '1990s': ('19900101', '19991231'),
        '2000s': ('20000101', '20091231'),
        '2010s': ('20100101', '20191231'),
    }

    for decade, tickers in tickers_by_decade.items():
        start_date, end_date = decade_dates.get(decade, (None, None))
        if start_date is None:
            continue
        output_path = os.path.join('output', decade)
        os.makedirs(output_path, exist_ok=True)
        pipeline.run(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_path=output_path
        )


if __name__ == '__main__':
    main()
