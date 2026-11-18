# Executar e testar o modelo unificado

Este roteiro substitui os comandos anteriores. Não há mais export-db ou SQLite no fluxo oficial. Use a raiz da cópia do repositório MRE.

## 1. Preparar o ambiente

Abra o Docker Desktop. No PowerShell:

```powershell
# Execute na raiz da sua cópia do repositório MRE.
uv sync --locked --dev
docker compose -f infra/docker-compose.yml up -d postgres
$env:DATABASE_URL = 'postgresql+psycopg2://finprov:finprov@localhost:5432/finprov'
uv run finprov db-init
uv run finprov db-status
```

db-init sem reset é idempotente no modelo novo. DATABASE_URL precisa ser definido em cada terminal. Em um banco novo, execute db-init antes de collect. Não use reset para atualizar os dados.

Para uma reinstalação deliberadamente destrutiva, o comando é `uv run finprov db-init --reset --confirm-database finprov`. Ele apaga somente os objetos conhecidos do projeto, recria os schemas e exige o nome exato do banco. Objetos desconhecidos/dependências externas impedem a exclusão; a operação é transacional. Não use para atualizar dados: use collect.

## 2. Consultar o resultado pronto ou coletar novamente

Sem consultar os sites:

```powershell
uv run finprov validate latest --recalculate
uv run finprov export latest --data-dir data/postgres
```

Espera-se status ok, problems vazio e risk_identical/direction_identical verdadeiros. O recálculo verifica os modelos atuais sobre curated e compara com os relatórios persistidos. Não reexecuta o parser nem consulta os sites. Confira também os alertas registrados em controle.qualidade.

Para uma nova coleta completa:

```powershell
uv run finprov collect --config config/carteira.toml
```

O intervalo padrão começa em 2015-01-01 e termina antes da data atual em São Paulo. Você pode fixar o fim exclusivo com --end AAAA-MM-DD. A fonte pode revisar valores mesmo com o mesmo intervalo.

O comando grava todas as fontes diretamente no PostgreSQL e termina em ready ou partial. Falhas inesperadas ficam failed. latest significa a última execução ready; use o UUID impresso para inspecionar uma coleta parcial. Uma chamada `--sources investidor10` serve como diagnóstico, mas nunca fica ready para o pipeline completo, pois não possui preços nem macro.

## 3. Abrir o dashboard oficial

No mesmo terminal com DATABASE_URL definido:

```powershell
Set-Location dash/motor-risco
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app.py --server.port 8502
```

Se a venv já existir, pule sua criação, mas atualize as dependências com uv pip install. Abra http://localhost:8502. O dashboard mantém as páginas originais do motor: Painel, VaR, ES, Volatilidade, Drawdown, Carteira, Stress e Backtesting. Selecione ativos e parâmetros na barra lateral. Os preços vêm da mesma execução ready do PostgreSQL, identificada na barra lateral; atualizar a página não faz nova coleta. Indicadores e demais dados tratados podem ser consultados no banco ou exportados pela CLI.

Não é necessário acesso ao Yahoo para abrir essa página com PostgreSQL disponível. Não há outra fonte de dados no dashboard: o acesso direto ao Yahoo foi removido. Ctrl+C encerra o servidor iniciado neste terminal. Se já houver uma instância em 8502, abra a existente ou escolha outra porta.

## 4. Consultar tabelas no PostgreSQL

Conexão no DBeaver/pgAdmin: host localhost, porta 5432, banco finprov, usuário finprov e senha finprov, conforme Compose local. Alternativa pelo terminal, a partir da raiz MRE:

```powershell
docker compose -f infra/docker-compose.yml exec postgres psql -U finprov -d finprov
```

No psql:

```sql
\dt raw.*
\dt curated.*
\dt controle.*
\dt resultados.*

SELECT run_id,status,started_at,finished_at FROM controle.execucoes;

SELECT ticker,COUNT(*) AS linhas,MAX(date) AS ultima_data
FROM curated.precos_atuais GROUP BY ticker ORDER BY ticker;

SELECT ticker,indicator,value,unit,period,reference_year
FROM curated.indicadores_atuais
WHERE ticker='VALE3' ORDER BY period,indicator,reference_year;

SELECT c.ticker,c.date,c.adjusted_close,r.source,r.url,r.observed_at
FROM curated.precos_atuais c JOIN raw.respostas r USING(raw_id,run_id)
WHERE c.ticker='AGRO3' ORDER BY c.date DESC LIMIT 5;

SELECT source,ticker,status,normalization_status,octet_length(body) AS bytes
FROM raw.respostas ORDER BY observed_at;

SELECT kind,payload->>'status' AS status FROM resultados.relatorios;
```

Para ler uma resposta textual, use `convert_from(body,'UTF8')` filtrando um raw_id específico. As respostas ficam em BYTEA para preservar seus bytes. Saia com `\q`.

Compare manualmente alguns indicadores atuais com as páginas indicadas em URL, levando em conta o horário da observação. Eventos de fontes diferentes não devem ser somados como se fossem pagamentos independentes. Percentuais dos indicadores usam pontos percentuais; risco usa fração do capital.

## 5. Testes automatizados

Na raiz MRE:

```powershell
uv run ruff check src tests
uv run pytest -q
```

Sem FINPROV_TEST_DATABASE_URL, os seis testes que exigem PostgreSQL são marcados como skipped. Para executá-los em banco separado e descartável:

```powershell
docker compose -f infra/docker-compose.yml exec -T postgres createdb -U finprov finprov_test_local
$env:FINPROV_TEST_DATABASE_URL = 'postgresql+psycopg2://finprov:finprov@localhost:5432/finprov_test_local'
uv run pytest -q
```

Crie esse banco uma única vez; createdb retorna erro se já existir. Os testes recusam nomes sem prefixo finprov_test_. Cada teste de integração reconstrói as estruturas desse banco descartável. Nunca aponte essa variável para finprov. A integração usa dados controlados e um banco separado; nenhum teste apaga o banco operacional.

Os dados artificiais são exclusivos dos testes e não entram na coleta real. Os testes cobrem fórmulas, prevenção de vazamento, parsing, 12 páginas independentes, falhas, transações, duplicações e preservação de objetos externos no reset.

Para testar a interface, instale pytest no ambiente do dashboard e execute a partir de dash/motor-risco:

```powershell
uv pip install --python .venv/Scripts/python.exe pytest
.venv/Scripts/python.exe -m pytest tests -q
```

Esses testes não exigem PostgreSQL ativo e bloqueiam HTTP para garantir que o dashboard não consulte o mercado.

## 6. Diagnóstico

- Conexão recusada: verifique Docker e serviço postgres. As portas de outros projetos não devem ser confundidas com 5432 do Compose deste repositório.
- Relação inexistente: execute db-init no banco correto. Ele não migra silenciosamente uma tabela antiga incompatível.
- Coleta partial: consulte controle.qualidade e raw.respostas pelo UUID. Corrija a causa e faça uma nova coleta; não remova ativos para esconder a falha.
- Modelo sem histórico: amplie o intervalo. O classificador exige pelo menos 504 linhas de treino, 252 de teste e observações para criar atributos.
- Recálculo diferente: verifique dados, código, versões e parâmetros; não sobrescreva o resultado original para encobrir a diferença.
- Arquivos antigos em data: não alimentam o novo ETL. Exporte do banco para data/postgres e use o UUID atual.

A comparação independente com planilha e a metodologia formal da monografia continuam necessárias para fechar todos os critérios acadêmicos da fase.
