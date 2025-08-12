# TEFAS Fund Analysis – August 2025

This project is an interactive **Streamlit dashboard** for analysing Turkish TEFAS mutual fund data.  
It brings together datasets from **TEFAS**, **Fonbul.com**, **TCMB inflation reports**, and other sources to give a clear view of fund performance.

### Features
- **Overview Page** – Summary stats and fund type distributions
- **Explore Funds** – Filter & search funds with sortable tables
- **Compare Funds** – Side-by-side performance & risk metrics
- **Risk / Return Analytics** – Scatterplots by Sharpe, return, and max loss
- **Market Benchmarks** – Gold & USD/TRY tracking
- **Inflation (TUFE)** – Inflation data & impact on returns
- **Top Funds by Risk Level** – Curated low, medium, and high-risk picks

### Data Sources
- [TEFAS](https://www.tefas.gov.tr/) – Fund performance and details  
- [Fonbul.com](https://www.fonbul.com/) – Risk metrics & fee data  
- [TCMB](https://www.tcmb.gov.tr/) – TUFE inflation data  
- [ExchangeRate.host](https://exchangerate.host/) – USD/TRY rates  
- [GoldAPI](https://www.goldapi.io/) – Gold prices

### Notes

    All data is for analysis purposes only – not financial advice.

    © Burak Ozteke, 2025. All rights reserved.

### How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
