# 123 Snake leaderboard server

Cloudflare Worker + Durable Objects + D1. One Durable Object per ranked
game holds the authoritative board via `engine.wasm` (the same engine the
lab embeds — `game_new` / `game_apply` / `game_dump` / `game_restore`);
clients only submit chain paths, so scores cannot be fabricated. Scores
live in D1; leaderboards are period queries (daily / monthly / all-time,
UTC). Also serves the site visit counter (`/api/hit`, daily uniques via
hashed ip+day — no raw IPs stored).

## API
- `POST /api/new` -> `{id, cells, score, over}` (rate limit 60 games/h/IP)
- `POST /api/move {id, path:[cell indices]}` -> `{cells, score, over}`
  (400 on illegal move; server refills from its own hidden RNG)
- `POST /api/finish {id, ini}` -> `{ok, score, daily_rank}`
  (only when `over`; `ini` = 3 letters, slur-blocked)
- `GET /api/board?p=daily|monthly|alltime` -> `{rows:[{ini,score,ts}]}`
- `POST /api/hit` -> `{total, today_unique}`

## Deploy (one-time)
1. Free Cloudflare account, then `npx wrangler login`.
2. `npx wrangler d1 create snake-scores` -> paste the printed
   `database_id` into wrangler.toml.
3. `npx wrangler d1 execute snake-scores --remote --file schema.sql`
4. `npx wrangler deploy` -> note the `*.workers.dev` URL, set it as
   `SNAKE_API` in the game page.

## Local dev
`npx wrangler dev` (uses a local D1; run the schema with
`npx wrangler d1 execute snake-scores --local --file schema.sql`).

## Updating the engine
`cargo build --profile wasm-release --target wasm32-unknown-unknown --lib`
then `cp target/wasm32-unknown-unknown/wasm-release/integer_snake.wasm
server/engine.wasm` and redeploy.

## Future (Labs roadmap, designed-for but not built)
Lookahead queue (expose next-N of the refill stream), 1v1 over
WebSockets on the same Durable Object (shared stream, per-move clock),
merge-value garbage injection into the opponent's queue.
