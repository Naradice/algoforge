// Canonical per-datasource-type config field definitions, shared between the create form
// (web/app/data/new/page.tsx) and the edit form (web/app/data/datasources/[id]/page.tsx).
//
// These two forms used to keep independent copies of this data, which drifted: the edit page's
// copy was missing `economic_calendar` entirely, `synthetic_function` entirely, and several
// `ddm_simulation` fields (model, spread, max_volatility, min_volatility, trade_unit). Since the
// edit form rebuilds `config` from *only* the fields it knows about (see `valuesToConfig` on the
// edit page), any field missing from its copy was silently dropped on save — for a type with no
// entry at all (economic_calendar, synthetic_function), *every* field was dropped, wiping config
// to `{}`. Single source of truth here so the two forms can't diverge again.

export interface FieldDef {
  key: string;
  label: string;
  type: "text" | "number" | "select" | "date";
  options?: string[];
  optionDescriptions?: Record<string, string>; // shown below select when an option is chosen
  placeholder?: string;
  hint?: string;
}

export const TIMEFRAME_OPTIONS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

export const TYPE_FIELD_DEFS: Record<string, FieldDef[]> = {
  ohlc_download: [
    {
      key: "client", label: "Provider", type: "select",
      options: ["yfinance", "vantage"],
      hint: "yfinance is free; vantage requires an API key",
    },
    {
      key: "symbol", label: "Symbol", type: "text",
      placeholder: "USDJPY=X",
      hint: "yfinance: USDJPY=X, EURUSD=X, AAPL. Alpha Vantage: USD/JPY",
    },
    { key: "timeframe", label: "Timeframe", type: "select", options: TIMEFRAME_OPTIONS },
    {
      key: "from_ts", label: "From Date", type: "date",
      hint: "yfinance H1/M data is limited to ~730 days lookback — older dates are auto-adjusted",
    },
    { key: "to_ts", label: "To Date", type: "date", placeholder: "", hint: "Leave blank for today" },
  ],
  ddm_simulation: [
    { key: "model", label: "Model Version", type: "select", options: ["v3", "v1"], hint: "V3 adds WMA trend-following feedback (original paper). V1 is the simpler base model — use to diagnose drift issues." },
    { key: "timeframe", label: "Timeframe", type: "select", options: TIMEFRAME_OPTIONS },
    { key: "initial_price", label: "Initial Price", type: "number", placeholder: "100.0" },
    { key: "spread", label: "Spread", type: "number", placeholder: "1.0", hint: "Bid-ask spread in price units" },
    { key: "num_agent", label: "Number of Agents", type: "number", placeholder: "300", hint: "More agents = lower volatility (original default: 300)" },
    { key: "max_volatility", label: "Max Volatility", type: "number", placeholder: "0.02", hint: "Upper bound of per-agent price tendency per step" },
    { key: "min_volatility", label: "Min Volatility", type: "number", placeholder: "0.01", hint: "Lower bound of per-agent price tendency per step" },
    { key: "trade_unit", label: "Trade Unit", type: "number", placeholder: "0.001", hint: "Minimum price increment (pips)" },
    { key: "seed", label: "Random Seed", type: "number", placeholder: "42" },
  ],
  web_report: [
    {
      key: "url", label: "URL", type: "text",
      placeholder: "https://www.example.com/reports",
      hint: "Landing page containing report links",
    },
    {
      key: "ext", label: "File Type", type: "select",
      options: ["pdf", "html", "mp3", "txt"],
    },
    {
      key: "subfolder", label: "Subfolder", type: "text",
      placeholder: "mizuho",
      hint: "Output directory name (under artifacts/web_reports/)",
    },
    {
      key: "filename", label: "Filename Template", type: "text",
      placeholder: "{YYYYMMDD}.pdf",
      hint: "Placeholders: {YYYYMMDD} {YYMMDD} {YYYYMM} {YYMM} {filename} {basefilename}",
    },
    {
      key: "type", label: "Fetch Method", type: "select",
      options: ["load", "goto_load", "goto_download", "load_rep"],
      optionDescriptions: {
        load:          "Direct HTTP download (httpx). Fast and simple — use for public links with no bot protection. Will get 403 on Akamai/CDN-protected sites.",
        goto_load:     "Opens the URL in a real browser and saves the rendered page as a PDF. Use when the target is an HTML page you want to archive as PDF, not a file download.",
        goto_download: "Uses the browser's fetch() to download the file in-page, carrying real browser headers and cookies. Required for Akamai/CDN-protected PDFs — most Japanese broker sites (Mizuho, Sony Finance, MUFG, etc.) need this.",
        load_rep:      "Plain HTTP download saved as HTML source. Use to archive a page's raw HTML. Fast, but shares the same CDN limitations as load — won't work on bot-protected sites.",
      },
    },
    {
      key: "unique", label: "Deduplication", type: "select",
      options: ["segment", "checksum", "text"],
      hint: "segment: skip if file exists. checksum: skip if content unchanged. text: skip if link text unchanged.",
    },
    {
      key: "interval_days", label: "Interval (days)", type: "number",
      placeholder: "1",
      hint: "Minimum days between downloads. Leave blank to always run.",
    },
    {
      key: "download_time", label: "Download time (UTC)", type: "text",
      placeholder: "18:00",
      hint: "Optional. Run at this time each day (HH:MM, UTC). If blank, runs immediately after interval_days.",
    },
  ],
  economic_calendar: [
    {
      key: "source", label: "Provider", type: "select",
      options: ["alpha_vantage", "fred"],
      optionDescriptions: {
        alpha_vantage: "Alpha Vantage economic indicators API. Requires a free API key from alphavantage.co.",
        fred: "Federal Reserve Economic Data (FRED). Requires a free API key from fred.stlouisfed.org.",
      },
    },
    {
      key: "api_key", label: "API Key", type: "text",
      placeholder: "your_api_key_here",
      hint: "Alpha Vantage: get free key at alphavantage.co/support/#api-key. FRED: fred.stlouisfed.org/docs/api/api_key.html",
    },
    {
      key: "indicators", label: "Indicators", type: "text",
      placeholder: "CPI,NONFARM_PAYROLL,UNEMPLOYMENT,FEDERAL_FUNDS_RATE",
      hint: "Alpha Vantage: CPI, NONFARM_PAYROLL, UNEMPLOYMENT, FEDERAL_FUNDS_RATE, REAL_GDP, RETAIL_SALES, DURABLES, TREASURY_YIELD, INFLATION. FRED: use series IDs e.g. CPIAUCSL, PAYEMS, UNRATE, FEDFUNDS, GDP.",
    },
    {
      key: "interval", label: "Interval", type: "select",
      options: ["monthly", "quarterly", "annual"],
      hint: "Alpha Vantage only. Controls the release frequency of the fetched series.",
    },
    { key: "from_ts", label: "From Date", type: "date" },
    { key: "to_ts", label: "To Date", type: "date", placeholder: "", hint: "Leave blank for today" },
  ],
  synthetic_function: [
    {
      key: "function", label: "Formula", type: "select",
      options: ["sine", "sine_sum", "delay", "xor", "lfsr"],
      optionDescriptions: {
        sine: "x_t = amplitude · sin(2π·t / period) — a single clean periodic wave.",
        sine_sum: "x_t = sin(2π·t / period) + amplitude · sin(2π·freq_ratio·t / period) — two frequencies mixed together.",
        delay: "Mackey-Glass delay-differential equation — the standard chaotic-time-series benchmark. Deterministic given tau, but long-range unpredictable in practice (sensitive to initial conditions). tau=17 is the canonical mildly-chaotic setting.",
        xor: "Temporal XOR: x_t = a(t-1) XOR a(t-2) for random bits a. Tests whether a model can learn a nonlinear (non-additive) temporal dependency, not just correlation/periodicity.",
        lfsr: "Linear feedback shift register — deterministic and simple to generate (one XOR per step, exact period 2^bits-1), but its statistics look close to random. Tests whether a model (or the token-characteristics framework) can tell 'looks complex' apart from 'is complex to generate'.",
      },
    },
    { key: "period", label: "Period (T, bars)", type: "number", placeholder: "50", hint: "sine / sine_sum only — bars per cycle of the base wave" },
    { key: "amplitude", label: "Amplitude (A)", type: "number", placeholder: "0.5", hint: "sine/sine_sum: wave amplitude. xor/lfsr: swing size (+/-A around base_price). Ignored by delay." },
    { key: "freq_ratio", label: "Frequency Ratio", type: "number", placeholder: "5", hint: "sine_sum only — how many times faster the 2nd wave oscillates than the base" },
    { key: "tau", label: "Delay (τ, bars)", type: "number", placeholder: "17", hint: "delay only — Mackey-Glass delay parameter. ~17 is mildly chaotic; below ~4.5 settles to a fixed point; higher values are more complex." },
    {
      key: "lfsr_bits", label: "Register Width (bits)", type: "select", options: ["4", "5", "8", "16"],
      hint: "lfsr only — sequence period is exactly 2^bits - 1 (e.g. 8 bits -> period 255)",
    },
    { key: "base_price", label: "Base Price", type: "number", placeholder: "100" },
    { key: "noise", label: "Noise (std dev)", type: "number", placeholder: "0", hint: "0 = pure deterministic signal" },
    { key: "length", label: "Length (bars)", type: "number", placeholder: "2000" },
    { key: "timeframe", label: "Timeframe", type: "select", options: TIMEFRAME_OPTIONS },
    { key: "start_ts", label: "Start Date", type: "date" },
    { key: "seed", label: "Random Seed", type: "number", placeholder: "42", hint: "Used by noise (all functions), and to generate bits for xor/lfsr. Unused by sine/sine_sum/delay, which are fully deterministic." },
  ],
  manual_upload: [],
};

export const TYPE_DEFAULTS: Record<string, Record<string, string>> = {
  ohlc_download: {
    client: "yfinance",
    symbol: "USDJPY=X",
    timeframe: "H1",
    from_ts: "2022-01-01",
    to_ts: "",
  },
  ddm_simulation: {
    model: "v3",
    timeframe: "M1",
    initial_price: "100",
    spread: "1",
    num_agent: "300",
    max_volatility: "0.02",
    min_volatility: "0.01",
    trade_unit: "0.001",
    seed: "42",
  },
  web_report: {
    url: "",
    ext: "pdf",
    subfolder: "",
    filename: "{YYYYMMDD}.pdf",
    type: "load",
    unique: "segment",
    interval_days: "1",
  },
  economic_calendar: {
    source: "alpha_vantage",
    api_key: "",
    indicators: "CPI,NONFARM_PAYROLL,UNEMPLOYMENT,FEDERAL_FUNDS_RATE",
    interval: "monthly",
    from_ts: "2020-01-01",
    to_ts: "",
  },
  synthetic_function: {
    function: "sine_sum",
    period: "50",
    amplitude: "0.5",
    freq_ratio: "5",
    tau: "17",
    lfsr_bits: "8",
    base_price: "100",
    noise: "0",
    length: "2000",
    timeframe: "M5",
    start_ts: "2024-01-01",
    seed: "42",
  },
  manual_upload: {},
};

export const TYPE_DESCRIPTIONS: Record<string, { label: string; description: string; status?: string }> = {
  ohlc_download: {
    label: "OHLC Download",
    description: "Download historical candle data from yfinance (free) or Alpha Vantage.",
  },
  ddm_simulation: {
    label: "DDM Simulation",
    description:
      "Generate synthetic OHLC data using a Deterministic Dealer Model. Tick data is stored and resampled to your chosen timeframe on demand.",
  },
  web_report: {
    label: "Web Report",
    description: "Download financial reports (PDF, HTML, audio) from institution websites using Playwright. Mirrors cyclic_downloader source.json schema.",
  },
  economic_calendar: {
    label: "Economic Calendar",
    description:
      "Download historical economic indicator releases (CPI, NFP, unemployment, Fed rate decisions) from Alpha Vantage or FRED. Stored as long-format parquet indexed by release date.",
  },
  synthetic_function: {
    label: "Synthetic Function",
    description:
      "Generate a time series from a closed-form formula or simple recurrence — sine/sine_sum for known periodicity, delay (Mackey-Glass) for chaotic-but-deterministic dynamics, xor for a nonlinear temporal dependency, or lfsr for a low-complexity generator that looks statistically random. The 'dataset axis' for comparing training/tokenization findings across generative rules of different character, not just different data sizes.",
  },
  manual_upload: {
    label: "Manual Upload",
    description:
      "Upload a CSV file from your computer as a dataset. The CSV must contain a 'close' column. Optionally include open, high, low, volume.",
  },
};
