# Fase 1: documentação vigente

O fluxo foi reconstruído em 18/11/2026. As instruções anteriores de coleta sem banco, export-db, SQLite e ETL legado não se aplicam mais.

- [Modelo unificado de dados](modelo-dados.md): tabelas, chaves, origem, datas, transações e limites.
- [Execução e testes](teste-fase1.md): PostgreSQL, coleta, validação, consultas e dashboard.

A coleta oficial persiste os brutos em raw, normaliza para curated e calcula os resultados a partir dessas tabelas. O dashboard usa preços dessa base para os cálculos interativos do motor de risco. Não há importação dos CSVs antigos.
