# AlgoForge Web

Next.js 14 dashboard for the AlgoForge platform.

## Requirements

- Node 20+
- Running AlgoForge backend (http://localhost:8000)

## Setup

```bash
npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL if backend is not on :8000
npm run dev
```

Open http://localhost:3000.

## Scripts

```bash
npm run dev      # development server with hot reload
npm run build    # production build
npm run start    # serve production build
npm run lint     # ESLint
```

## Project structure

```
web/
├── app/                      Next.js App Router
│   ├── layout.tsx            Root layout — sidebar + page wrapper
│   ├── globals.css           Tailwind base + custom token colours
│   ├── page.tsx              Redirects to /dashboard
│   ├── dashboard/
│   │   └── page.tsx          Overview — recent datasets, runs, models
│   ├── data/
│   │   ├── page.tsx          Dataset + datasource list
│   │   ├── new/page.tsx      Create datasource (OHLC / DDM / web report)
│   │   ├── datasources/[id]/page.tsx  Datasource detail + run collection job
│   │   └── datasets/[id]/page.tsx     Dataset detail + preview + characteristics
│   ├── model/
│   │   ├── page.tsx          Model list
│   │   ├── new/page.tsx      Create model (architecture + config JSON)
│   │   └── [id]/page.tsx     Model detail — training runs, deploy, validations
│   └── strategy/
│       ├── page.tsx          Strategy list
│       ├── new/page.tsx      Create strategy (definition JSON with templates)
│       └── [id]/page.tsx     Strategy detail — runs, metrics, trades, AI chat
├── components/
│   ├── sidebar.tsx           Navigation sidebar
│   └── status-badge.tsx      Colour-coded status pill
└── lib/
    └── fetcher.ts            SWR fetcher — wraps fetch with error handling
```

## Key patterns

### Data fetching

All data fetching uses [SWR](https://swr.vercel.app/). The `fetcher` from `lib/fetcher.ts` throws on non-2xx responses so SWR surfaces errors correctly.

```tsx
const { data, isLoading, error } = useSWR("/api/v1/models", fetcher);
```

For live-updating data (runs in progress), pass `refreshInterval`:

```tsx
const { data: runs } = useSWR(`/api/v1/strategies/${id}/runs`, fetcher, {
  refreshInterval: 3000,
});
```

After a mutation, call `mutate(key)` to revalidate:

```tsx
await fetch("/api/v1/models", { method: "POST", ... });
mutate("/api/v1/models");
```

### API proxy

Next.js rewrites `/api/v1/*` → `http://backend:8000/api/v1/*` in `next.config.js`. This means all `fetch` calls in components use relative paths — no hardcoded backend URL.

### WebSocket (chat)

The strategy detail page opens a WebSocket to `ws://localhost:8000/api/v1/ws/strategies/{id}/runs/{run_id}/chat`.

```tsx
const ws = new WebSocket(`ws://...`);
ws.onmessage = (e) => {
  const { role, content, is_final } = JSON.parse(e.data);
  // append to messages state
};
ws.send(JSON.stringify({ message: "How is the strategy performing?" }));
```

### Strategy definition templates

The strategy creation form (`/strategy/new`) includes three pre-filled templates selectable via dropdown:
- **MACD + RSI** — standard comparison conditions
- **ML Signal** — uses a deployed ML model for entry/exit
- **LLM Signal** — uses Gemini API to generate directional signals

## Styling

[Tailwind CSS](https://tailwindcss.com/) with a custom `brand-500` colour token (sky blue `#0ea5e9`).

Dark theme throughout. Background `gray-950`, panels `gray-900`, borders `gray-800`.

Status badges use the `StatusBadge` component which maps status strings to colour classes:

| Status | Colour |
|--------|--------|
| active / deployed / completed / ready | green |
| running / training | blue (animated pulse) |
| pending | yellow |
| error | red |
| stopped / inactive / archived | gray |

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend base URL (used for rewrites) | `http://localhost:8000` |
| `NEXT_PUBLIC_WS_URL` | WebSocket base URL | `ws://localhost:8000` |
