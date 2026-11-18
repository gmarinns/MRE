# Modelo unificado de dados e decisões da reconstrução

Versão 1 do esquema, implementada em 18/11/2026. Este documento substitui as instruções dos dois ETLs anteriores. O PostgreSQL é a fonte dos dados e dos resultados do fluxo oficial.

## 1. O que mudou

Foram excluídos o ETL `etl/precos.py` e as tabelas antigas `raw.cotacoes`, `curated.precos`, `public.finprov_records` e `public.finprov_runs` do banco local `finprov`. Os schemas do projeto foram reconstruídos. A nova tabela `curated.precos` tem outro contrato; não é a tabela antiga reaproveitada. O schema padrão public foi preservado, sem tabelas do FinProv. Outros bancos e serviços não fazem parte dessa exclusão.

Há um único comando de coleta. Não existe mais o caminho de calcular tudo em arquivos e depois importar JSON genérico para public. CSV/JSON/Markdown são exportações opcionais para consulta, não a origem dos cálculos.

```text
Fontes → raw.respostas → normalização → curated → modelos → resultados
              controle.execucoes identifica todo o percurso
```

Cada resposta é gravada no PostgreSQL antes de ser interpretada. O normalizador relê os bytes de raw; depois os modelos releem as tabelas de curated. O dashboard lê os preços de uma única execução ready e alimenta o motor de risco interativo. Esses cálculos não sobrescrevem os relatórios do ETL.

## 2. Camadas e tabelas

| Tabela | Conteúdo e finalidade |
|---|---|
| `controle.esquema` | Versão do modelo instalado |
| `controle.execucoes` | UUID, início/fim, status, parâmetros, ambiente e erro da execução |
| `controle.posicoes` | Os ativos e pesos de cada execução |
| `raw.respostas` | Resposta HTTP original em BYTEA, fonte, URL, parâmetros, horário e estado da normalização |
| `curated.ativos` | Cadastro de 12 ações e do benchmark, com símbolo do Yahoo |
| `curated.precos` | OHLC, volume e fechamento ajustado em colunas tipadas |
| `curated.eventos` | Dividendos, JCP, desdobramentos e demais eventos com suas datas |
| `curated.indicadores` | Indicadores atuais e anuais de cada empresa |
| `curated.macro` | Observações da Selic, dólar e IPCA |
| `controle.qualidade` | Contagens, erros de normalização, flags e problemas de completude |
| `controle.cobertura` | Cobertura de fontes por ação |
| `resultados.relatorios` | Resultados de risco e direção, ligados à execução |

Os dados de domínio ficam tabulares em curated; estruturas variáveis de parâmetros, diagnóstico e relatórios ficam em JSONB. Não é preciso transformar vetores de previsões e configurações em centenas de colunas nesta fase. Na Fase 2, previsões diárias devem ganhar tabela própria, a ser especificada na integração do motor.

Uma única tabela raw é suficiente porque seu contrato é o de uma resposta de fonte, não o de uma cotação. BYTEA preserva o HTML/JSON recebido sem perder bytes; source identifica o parser. O conteúdo é o corpo disponibilizado pelo cliente HTTP, não uma captura de pacotes ou arquivo completo de cabeçalhos HTTP.

## 3. Granularidade e chaves

- Preços: uma observação por execução, ativo e data. A chave impede duplicações dentro da execução.
- Indicadores: uma observação por execução, ativo, nome, período e ano de referência. `current` não tem ano; `annual` exige ano. Cada indicador tem coluna numérica value e coluna unit.
- Macro: uma observação por execução, série e data. IPCA continua mensal; não é artificialmente espalhado em dias.
- Eventos: uma linha por resposta e posição na extração. Fontes continuam separadas; não se presume que duas fontes descrevendo um pagamento representam dois pagamentos.
- Carteira: uma linha por execução e ativo. Pesos finitos, não negativos, somando 1 são validados antes da carga.

Cada linha tratada possui `run_id` e `raw_id`. Uma chave estrangeira composta impede referenciar um bruto de outra execução. Ativos têm chave estrangeira para o cadastro. Valores financeiros usam NUMERIC sem arredondamento fixo na persistência; os modelos recebem float64. Esta escolha não transforma valores fornecidos como float pelo Yahoo em uma precisão econômica superior à fonte.

Não existe uma tabela gigante juntando preços e fundamentos. As frequências são diferentes; a preparação dos modelos fará a associação temporal somente quando as datas de disponibilidade forem confiáveis.

## 4. Datas, proventos e qualidade

`date` é a data econômica da observação; `observed_at` é o horário da coleta em UTC. Para macro e fundamentos, `available_at` registra quando o sistema observou o valor, sem fingir ser a primeira divulgação pública. `published_at` permanece nulo quando desconhecido. Fundamentos têm `point_in_time_verified=false`.

Os eventos preservam data-com, data-ex e pagamento nos campos efetivamente fornecidos. Não se inventa data-ex como data-com mais um dia corrido. Dividendos e JCP são distinguidos quando a fonte permite. Bonificação não vira pagamento em dinheiro nem desdobramento com fator adivinhado. Os eventos não são somados novamente ao retorno que já usa preço ajustado.

Nulos não viram zero. Preços inválidos viram nulos sinalizados; o valor original permanece em raw. Falhas de HTTP ou layout ficam identificadas. Se uma ação não estiver disponível, não se redistribui seu peso pelas demais. Um indicador negativo não é automaticamente inválido: P/L negativo pode ser informativo.

## 5. Execuções e falhas

O ciclo normal é `collecting → processing → ready`. Uma coleta com fontes, fundamentos ou modelos incompletos termina em `partial`; uma falha inesperada termina em `failed`. Não há promoção silenciosa de dados incompletos.

Cada resposta tem seu próprio commit. Assim, uma falha posterior não perde os brutos já recebidos. A carga de curated ocorre em uma transação: uma violação de chave ou restrição reverte essa carga. Resultados, diagnóstico e status final também são gravados juntos. Uma queda abrupta do processo pode deixar collecting/processing; essas execuções não entram nas views de pronta utilização.

`latest` no banco significa a execução **ready** mais recente. As views `curated.precos_atuais` e `curated.indicadores_atuais` apontam para essa mesma execução, não para o último registro independente de cada ativo. Uma coleta parcial nova não substitui a anterior.

## 6. Retirada dos snapshots e preservação do RAW

A versão anterior deixou de salvar snapshots em disco a pedido do autor. O pedido posterior passou a exigir RAW e curated no banco. Portanto, agora preservamos as respostas originais por coleta **no PostgreSQL**. Isso é preservação de versões dos brutos; não deve ser descrito como ausência de histórico.

Não foi restaurada a antiga pasta de objetos SHA-256 nem a cópia automática de código. `data/` pode conter relatórios antigos, mas o novo pipeline não os importa ou utiliza. Backups do banco passam a ser necessários para conservar os brutos e a linhagem. O registro de commit-base não identifica mudanças locais não commitadas: a versão definitiva dos experimentos deve ser registrada no Git.

## 7. Uso no modelo e no dashboard

O risco e o classificador leem preços da mesma execução em curated. A carteira usa 1/12 do valor em cada ação, com hipótese de pesos constantes/rebalanceamento diário, sem custos. Há VaR histórico, ES empírico, volatilidade, drawdown e backtesting descritivo. Direção de próximo pregão é um experimento separado, ainda sem demonstração de vantagem preditiva.

Fundamentos e macro estão preparados e consultáveis, mas não entram automaticamente no classificador. Não se aplica um valor atual a datas passadas para fabricar uma série histórica. Isso provocaria vazamento.

O dashboard preserva o visual, as páginas e os cálculos do pacote motor_calculo. Apenas a origem das cotações mudou: curated.precos da mesma execução ready para todos os ativos, sem coleta direta do Yahoo ou fallback em disco. O adaptador aplica adjusted_close / close ao OHLC e preserva o volume. Os períodos terminam na última data da execução.

Os resultados interativos não são necessariamente iguais aos relatórios persistidos: o motor usa log-retornos e parâmetros da interface; o relatório do ETL usa retornos simples e sua janela configurada. Eles compartilham a origem dos preços, mas são cálculos distintos. O grafo W3C PROV da Fase 3 ainda não está implementado.

## 8. Verificação e limites

A suíte em tests cobre parsing, fórmulas, prevenção de vazamento, pesos, falhas de fonte, integridade e transações no PostgreSQL. A suíte em dash/motor-risco/tests verifica a interface com consultas HTTP bloqueadas, incluindo as nove páginas com o motor real, ajuste de preços, parâmetros, configuração ausente e banco indisponível. O CI executa a integração em um banco PostgreSQL descartável.

Use validate --recalculate para conferir uma execução real. Isso compara resultados calculados sobre curated com os persistidos; não reexecuta o parser nem comprova veracidade da fonte. Consulte [teste-fase1.md](teste-fase1.md) para os comandos.
