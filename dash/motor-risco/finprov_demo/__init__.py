"""Pacote de apoio da tela de proposta do FinProv (Streamlit).

A separação em módulos espelha, de propósito, a arquitetura definida na
proposta do Motor de Risco:

    finprov_demo/risk.py         ->  src/finprov/risk/{var,es,volatility,portfolio}.py
    finprov_demo/backtesting.py  ->  src/finprov/risk/backtesting.py
    finprov_demo/data.py         ->  src/finprov/etl/{extract_yahoo,transform}.py
    finprov_demo/provenance.py   ->  src/finprov/provenance/tracker.py
    finprov_demo/charts.py       ->  dash/components/charts.py

Assim, a migração para o repositório MRE é um *move* de arquivo, não uma
reescrita.
"""

__version__ = "0.2.0"
