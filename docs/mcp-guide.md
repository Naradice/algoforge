# MCP Guide — AI Agent Integration

AlgoForge exposes all three layers as MCP (Model Context Protocol) tools. An AI agent (Claude, GPT-4, etc.) can autonomously manage the full trade development loop: collect data → train model → build strategy → backtest → iterate.

## Connecting

The MCP server is mounted at `http://localhost:8000/mcp` using SSE transport. Add to your Claude Desktop or agent configuration:

```json
{
  "mcpServers": {
    "algoforge": {
      "url": "http://localhost:8000/mcp",
      "transport": "sse"
    }
  }
}
```

---

## Available Tools

### Data Tools

| Tool | Description |
|------|-------------|
| `list_datasources()` | List all datasources with collection job status |
| `create_datasource(name, type, config)` | Create a new datasource |
| `collect_data(datasource_id)` | Trigger data collection |
| `list_datasets()` | List all datasets with metadata |
| `get_dataset_download(dataset_id)` | Get artifact path and download URL for the full dataset |
| `get_dataset_preview(dataset_id, rows=5)` | Get first/last N rows |
| `get_dataset_characteristics(dataset_id)` | Get Hurst, ACF, kurtosis, etc. with interpretation |
| `analyze_dataset(dataset_id)` | Compute dataset characteristics |
| `get_dataset_info(dataset_id)` | Full dataset metadata |

### Model Tools

| Tool | Description |
|------|-------------|
| `list_models()` | List all models with status |
| `create_model(name, architecture, config)` | Create a model |
| `list_preprocessed_datasets(dataset_id=None)` | List saved preprocessing recipes (optionally for one dataset) |
| `get_preprocessed_dataset(preprocessed_dataset_id)` | Recipe config + its structure characteristics |
| `start_training_run(model_id, hyperparams, dataset_id=None, preprocessed_dataset_id=None, execution_target="local")` | Start training — prefer `preprocessed_dataset_id` when a recipe exists. `execution_target="colab"` runs on a Google Colab CPU runtime instead of this backend's own worker (see [colab-workflow.md](colab-workflow.md)); only `architecture="lstm"` with no recipe/token_level/preprocessing and `split_mode="chronological"` is supported for it today |
| `get_training_status(training_run_id)` | Poll status with epoch/ETA |
| `stop_training_run(training_run_id)` | Gracefully stop training |
| `get_model_training_runs(model_id)` | List training run history |
| `get_model_validations(model_id)` | Get validation metrics |
| `compare_model_runs(model_id)` | Compare runs ranked by val_loss |
| `deploy_model(model_id, training_run_id)` | Deploy best run |
| `predict(model_id, features, feature_names)` | Run inference |
| `start_hyperparameter_search(model_id, dataset_id, search_grid, execution_target="local")` | Grid search — `execution_target="colab"` runs every combination on Colab (see [colab-workflow.md](colab-workflow.md)); actual concurrency depends on the `colab` queue's worker count |

Preprocessed datasets aren't created via MCP yet (no `create_preprocessed_dataset` tool) — create
one through the UI (`/data/preprocessed/new`) or `POST /preprocessed-datasets`, then reference it
by ID from `start_training_run`.

### Strategy Tools

| Tool | Description |
|------|-------------|
| `list_strategies()` | List all strategies with last run summary |
| `get_strategy_definition(strategy_id)` | Full strategy JSON |
| `create_strategy(name, definition, description)` | Create a strategy |
| `update_strategy(strategy_id, definition)` | Update strategy (auto-versions) |
| `start_strategy_run(strategy_id, mode, dataset_id, broker_client, from_ts, to_ts)` | Start run |
| `get_run_status(strategy_id, run_id)` | Poll status and progress |
| `get_run_metrics(run_id)` | Performance metrics with interpretation |
| `get_run_trades(run_id, limit)` | Individual trades |
| `compare_runs(strategy_id, run_ids)` | Compare multiple runs |
| `stop_strategy_run(strategy_id, run_id)` | Stop a run |
| `send_strategy_chat(strategy_id, run_id, message)` | Chat with the run AI |

### Log Tools

| Tool | Description |
|------|-------------|
| `get_run_logs(run_id, run_type, level, limit)` | Logs for a run |
| `search_logs(query, level, limit)` | Full-text log search |
| `get_log_summary(run_id, run_type)` | Error summary |

---

## MCP Resources

Resources expose structured read-only data without tool calls:

| URI | Content |
|-----|---------|
| `algoforge://strategies/{id}` | Strategy definition as JSON |
| `algoforge://strategies/{id}/runs/{run_id}/metrics` | Run metrics as JSON |
| `algoforge://datasets/{id}/characteristics` | Dataset characteristics |
| `algoforge://dashboard` | Platform-wide summary |

---

## Example Agent Workflows

### Automated training loop
```
1. list_datasets()                                        → find a ready dataset
2. list_preprocessed_datasets(dataset_id)                 → reuse an existing recipe if one exists
3. create_model("LSTM v1", "lstm", {...})                 → create model
4. start_training_run(model_id, hp, preprocessed_dataset_id=recipe_id)   → start training
5. loop: get_training_status(run_id)                      → wait for completion
6. get_model_validations(model_id)                        → check val metrics
7. if val_loss > threshold: update hyperparams, goto 4
8. deploy_model(model_id, best_run_id)
```

### Automated backtest iteration
```
1. get_strategy_definition(strategy_id)             → read current definition
2. start_strategy_run(id, "backtest", dataset_id)   → run backtest
3. get_run_metrics(run_id)                          → read performance
4. send_strategy_chat(id, run_id, "How to improve win rate?")
5. update_strategy(id, improved_definition)         → apply changes
6. goto 2                                           → iterate
```

### Data quality assessment
```
1. list_datasets()                          → find dataset
2. get_dataset_characteristics(dataset_id)  → check Hurst, kurtosis
3. get_dataset_preview(dataset_id)          → inspect values
4. if hurst < 0.45: note "mean-reverting, use RSI strategies"
5. if kurtosis > 5: note "fat tails, use conservative position sizing"
```

---

## Webhook Events

Register a webhook to receive push notifications instead of polling:

```
POST /webhooks
{
  "url": "https://your-agent-endpoint/hook",
  "events": ["run.completed", "run.error", "training.completed", "collection.completed"],
  "secret": "your-hmac-secret"
}
```

Payload is HMAC-SHA256 signed with the secret in `X-AlgoForge-Signature` header. Verify before processing.

Event types: `run.completed`, `run.error`, `training.completed`, `training.error`, `collection.completed`, `collection.error`.
