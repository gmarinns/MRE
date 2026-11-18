# Dashboard FinProv

Interface original do motor de risco, com navegação, gráficos, tema e cálculos preservados. O adaptador `finprov_dashboard/dados.py` lê somente o PostgreSQL; `finprov_dashboard/servicos.py` continua chamando `motor_calculo`. Não há coletor Yahoo ou fallback em arquivos no dashboard.

No PowerShell, nesta pasta:

```powershell
$env:DATABASE_URL = 'postgresql+psycopg2://finprov:finprov@localhost:5432/finprov'
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app.py --server.port 8502
```

Se a venv já existir, pule sua criação, mas atualize as dependências. Abra http://localhost:8502. Consulte [o roteiro de execução](../../docs/teste-fase1.md) para preparar e popular o banco.

Cada renderização fixa a última execução ready; todos os ativos usam esse mesmo run_id, mostrado na barra lateral. O histórico de 1, 2 ou 5 anos termina na última data de preços dessa execução. O cache de consulta dura 30 segundos e inclui o run_id na chave. Atualizar a página consulta o banco novamente, sem coletar dados externos.

O fechamento ajustado e o fator adjusted_close / close ajustam OHLC para manter o contrato anterior do motor. O volume não é ajustado. As 12 ações da carteira ficam selecionadas por padrão, com pesos iguais na página Carteira; os controles originais permitem explorar outras configurações.

Os cálculos do painel são interativos e não sobrescrevem resultados.relatorios. Eles podem diferir dos relatórios do ETL: o motor usa log-retornos e o período escolhido; o relatório persistido usa retornos simples e sua própria janela. Fundamentos, macro, previsão e qualidade continuam disponíveis no banco e na CLI; não foi criada uma nova interface para eles.

Sem PostgreSQL acessível ou execução pronta, a interface informa o problema e não busca outra fonte. A suíte `tests` executa as nove páginas com o motor real e HTTP bloqueado.
