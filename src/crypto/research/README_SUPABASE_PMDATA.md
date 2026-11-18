## Windows PowerShell / local venv environment fix

Install `python-dotenv` alongside the other dependencies:

```powershell
python -m pip install python-dotenv boto3 requests pandas pyarrow
```

The three scripts in this updated bundle now load `.env` from the repository root automatically, even when run directly with `python src/crypto/research/...`. Environment variables already set in the running process take precedence (`override=False`). This does not create S3 credentials: `PREDICTION_DATABASE_URL` is for PostgreSQL and cannot replace the four `SUPABASE_S3_*` settings.

On local Windows, to migrate files that exist in the local cache:

```powershell
python -u -B src/crypto/research/pmdata_supabase_storage.py --dataset btc-1h --migrate-cache data/research/poly_btc1h_4545/cache
```

If cache files are still only in Codespaces, the migration cannot see them. Run a backtest with `--storage supabase` to stream uncached data from PMData directly to Supabase instead.

---

# Polymarket PMData backtests — Supabase Storage edition

These three Python scripts belong side-by-side in `src/crypto/research/`:

- `pmdata_supabase_storage.py` — streams raw PMData response objects into Supabase Storage via S3 multipart uploads, and reads ZIP/Parquet contents using S3 HTTP byte ranges.
- `polymarket_btc1h_4545_backtest.py` — hourly BTC 45¢/45¢ backtest.
- `polymarket_btc5m_4545_backtest.py` — five-minute BTC 45¢/45¢ backtest, sharing the hourly execution proxy.

## First-time setup (no keys in chat, shell history, or repo)

1. In your existing **Supabase** project, open Storage and create a **private** file bucket called `prediction-market-data`. Make sure your project's maximum file size permits the largest compressed daily ZIP you intend to store. Supabase Free projects cap individual objects at 50 MB; a large daily ZIP may require an upgraded project or choosing individual-market Parquet files.
2. In Supabase **Storage → Settings → S3 protocol**, enable S3 and generate a server-side S3 credential pair. Take the S3 endpoint and region from the Supabase dashboard.
3. Save these values as **GitHub Codespaces secrets** (not plaintext shell `export` commands):
   - `SUPABASE_S3_ENDPOINT` (often `https://<project-ref>.storage.supabase.co/storage/v1/s3`)
   - `SUPABASE_S3_REGION` (copy the region displayed in Storage S3 settings)
   - `SUPABASE_S3_ACCESS_KEY_ID`
   - `SUPABASE_S3_SECRET_ACCESS_KEY`
   - `PMDATA_API_KEY` (your existing PMData credential)
4. Rebuild/restart the Codespace so these secrets appear in its environment.
5. Run `python3 -m pip install requests pandas pyarrow boto3`.

Your existing `PREDICTION_DATABASE_URL` authorizes **PostgreSQL**, *not* object Storage. S3 needs a separate credential pair. Don't share credentials. Supabase S3 keys are privileged server-side credentials; use only trusted server-side processes.

## Migration of already downloaded hourly ZIP archives (no re-download)

```bash
cd /workspaces/prediction-market-bot
python3 -B src/crypto/research/pmdata_supabase_storage.py --self-test
python3 -u -B src/crypto/research/pmdata_supabase_storage.py \
  --dataset btc-1h \
  --migrate-cache data/research/poly_btc1h_4545/cache
```

The original files stay intact. After verifying the uploads in the Supabase Storage dashboard, you can manually remove the old local `cache/` files to reclaim Codespaces disk. Don't delete during a running backtest.

For any previously downloaded 5m samples (optional):

```bash
python3 -u -B src/crypto/research/pmdata_supabase_storage.py \
  --dataset btc-5m \
  --migrate-cache data/research/poly_btc5m_4545/cache
```

## Test the 5-minute 45/45 strategy from Supabase

```bash
python3 -B src/crypto/research/polymarket_btc5m_4545_backtest.py --self-test
python3 -u -B src/crypto/research/polymarket_btc5m_4545_backtest.py \
  --storage supabase \
  --date 2026-10-12 --start-hour 12 --hours 2 \
  --source individual --price 0.45 --size 5 --latency-ms 250
```

For 24 hours of 5-minute markets, change to `--start-hour 0 --hours 24`. Use `--source daily` only if your Supabase Storage project permits the size of the 3 daily ZIPs. Each already archived object is reused without re-downloading from PMData.

## Re-run hourly 45/45 using cloud archive

```bash
python3 -u -B src/crypto/research/polymarket_btc1h_4545_backtest.py \
  --storage supabase --offline \
  --start 2026-10-10 --end 2026-10-16 \
  --price 0.45 --size 5 --latency-ms 250
```

`--offline` prevents any PMData downloads; it requires every archive needed for that period already exists in Supabase. Omit `--offline` if you want to fetch missing archives.

**Cloud behavior**: PMData sends bytes through Codespaces *memory* to Supabase; no full ZIPs/Parquet files are written to the Codespaces filesystem. Backtesting reads only needed members from cloud ZIPs via range requests. Small CSV/JSON output files are written locally *and* copied to Supabase under `research/polymarket/btc-1h/` or `research/polymarket/btc-5m/`.

**Important caveats**: This changes storage, **not** the original fill model: 45/45 order touch and print/queue proxies are not confirmed live maker fills. Network transfer through Codespaces still consumes bandwidth. Supabase egress, S3 API requests, and storage consumption may cost money. Loading dozens of market objects remotely may be slower than the former local cache. Replays still repeatedly reprocess L2 for every limit price; multi-price batched replay is a separate optimization.
