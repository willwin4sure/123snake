// 123 Snake leaderboard API — Cloudflare Worker + Durable Objects + D1.
//
// Server-authoritative ranked games: the Durable Object holds the real
// board (via the same wasm engine the lab uses), the client only submits
// chain paths. Scores land in D1; leaderboards are period queries.

import ENGINE from "./engine.wasm";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "content-type",
};

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", ...CORS },
  });

// Display-name filter: hard slurs are substring-matched after leetspeak
// normalization (these are never innocently embedded); milder terms are
// whole-string only, to avoid Scunthorpe-style false positives.
const LEET = { "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s", "!": "i" };
const SLUR_SUBSTR = [
  "nigg", "niger", "fagg", "kike", "spic", "wetback", "chink", "tranny",
  "retard", "raghead", "beaner", "gook", "coon",
];
const BAD_EXACT = new Set([
  "hitler", "nazi", "kkk", "rape", "rapist", "cunt", "faggot",
]);
function nameAllowed(name) {
  const norm = name.toLowerCase().split("").map((ch) => LEET[ch] || ch).join("")
    .replace(/[^a-z]/g, "");
  if (SLUR_SUBSTR.some((w) => norm.includes(w))) return false;
  if (BAD_EXACT.has(norm)) return false;
  return true;
}

// IP pseudonymization: HMAC-SHA256 with a server-side salt when the
// IP_SALT secret is set (wrangler secret put IP_SALT), so stored hashes
// cannot be reversed by brute-forcing the IPv4 space. Falls back to a
// plain digest in local dev.
async function iphash(env, s) {
  const enc = new TextEncoder();
  if (env.IP_SALT) {
    const key = await crypto.subtle.importKey(
      "raw", enc.encode(env.IP_SALT), { name: "HMAC", hash: "SHA-256" },
      false, ["sign"]
    );
    const d = await crypto.subtle.sign("HMAC", key, enc.encode(s));
    return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
  }
  const d = await crypto.subtle.digest("SHA-256", enc.encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const p = url.pathname;
    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }

    if (p === "/api/new" && req.method === "POST") {
      const ip = req.headers.get("cf-connecting-ip") || "0";
      const iph = await iphash(env, ip);
      const hourAgo = Date.now() - 3600_000;
      const row = await env.DB.prepare(
        "SELECT COUNT(*) AS cnt FROM newlog WHERE iph=? AND ts>?"
      ).bind(iph, hourAgo).first();
      if (row.cnt >= 60) return json({ error: "rate limited" }, 429);
      await env.DB.prepare("INSERT INTO newlog(ts, iph) VALUES(?,?)")
        .bind(Date.now(), iph).run();
      const id = crypto.randomUUID().replaceAll("-", "");
      const stub = env.GAME.get(env.GAME.idFromName(id));
      const r = await stub.fetch("https://do/new", { method: "POST" });
      const state = await r.json();
      return json({ id, ...state });
    }

    if (p === "/api/move" && req.method === "POST") {
      const body = await req.json().catch(() => ({}));
      const { id, path, n } = body;
      if (typeof id !== "string" || !/^[0-9a-f]{32}$/.test(id) || !Array.isArray(path)) {
        return json({ error: "bad request" }, 400);
      }
      const stub = env.GAME.get(env.GAME.idFromName(id));
      const r = await stub.fetch("https://do/move", {
        method: "POST",
        body: JSON.stringify({ path, n }),
      });
      return json(await r.json(), r.status);
    }

    if (p === "/api/finish" && req.method === "POST") {
      const body = await req.json().catch(() => ({}));
      const { id, name } = body;
      if (typeof id !== "string" || !/^[0-9a-f]{32}$/.test(id)) {
        return json({ error: "bad request" }, 400);
      }
      const clean = String(name || "").replace(/[^\w .-]/g, "").replace(/\s+/g, " ").trim().slice(0, 16);
      if (clean.length < 2) return json({ error: "name must be 2-16 characters" }, 400);
      if (!nameAllowed(clean)) return json({ error: "pick a different name" }, 400);
      const stub = env.GAME.get(env.GAME.idFromName(id));
      const r = await stub.fetch("https://do/finish", { method: "POST" });
      if (!r.ok) return json(await r.json(), r.status);
      const fin = await r.json();
      await env.DB.prepare(
        "INSERT INTO scores(ts, name, score, moves) VALUES(?,?,?,?)"
      ).bind(Date.now(), clean, fin.score, fin.moves).run();
      const better = await env.DB.prepare(
        "SELECT COUNT(*) AS n FROM scores WHERE score>? AND ts>=?"
      ).bind(fin.score, Date.UTC(new Date().getUTCFullYear(), new Date().getUTCMonth(), new Date().getUTCDate())).first();
      return json({ ok: 1, score: fin.score, daily_rank: better.n + 1 });
    }

    if (p === "/api/state" && req.method === "GET") {
      const id = url.searchParams.get("id") || "";
      if (!/^[0-9a-f]{32}$/.test(id)) return json({ error: "bad request" }, 400);
      const stub = env.GAME.get(env.GAME.idFromName(id));
      const r = await stub.fetch("https://do/state");
      return json(await r.json(), r.status);
    }

    if (p === "/api/board" && req.method === "GET") {
      const period = url.searchParams.get("p") || "daily";
      const now = new Date();
      let since = 0;
      if (period === "daily") {
        since = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
      } else if (period === "monthly") {
        since = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1);
      }
      // every finished game is its own immutable row (arcade style):
      // same-name submissions add entries, they never replace anyone's
      const { results } = await env.DB.prepare(
        "SELECT name, score, ts FROM scores WHERE ts>=? ORDER BY score DESC, ts ASC LIMIT 10"
      ).bind(since).all();
      return json({ rows: results });
    }

    if (p === "/api/hit" && req.method === "POST") {
      const ip = req.headers.get("cf-connecting-ip") || "0";
      const day = new Date().toISOString().slice(0, 10);
      const iph = await iphash(env, ip + day);
      // per-visitor daily hit cap so the total can't be curl-spammed
      await env.DB.prepare(
        "INSERT INTO hits(day, iph, n) VALUES(?,?,1) ON CONFLICT(day, iph) DO UPDATE SET n=n+1"
      ).bind(day, iph).run();
      const mine = await env.DB.prepare(
        "SELECT n FROM hits WHERE day=? AND iph=?"
      ).bind(day, iph).first();
      if (mine.n <= 100) {
        await env.DB.prepare(
          "INSERT INTO meta(k, v) VALUES('total', 1) ON CONFLICT(k) DO UPDATE SET v=v+1"
        ).run();
      }
      const total = await env.DB.prepare("SELECT v FROM meta WHERE k='total'").first();
      const uniq = await env.DB.prepare(
        "SELECT COUNT(*) AS u FROM hits WHERE day=?"
      ).bind(day).first();
      return json({ total: total ? total.v : 0, today_unique: uniq.u });
    }

    if (p === "/") return new Response("123 snake api", { headers: CORS });
    return json({ error: "not found" }, 404);
  },
};

// One Durable Object per ranked game: owns the authoritative board via the
// wasm engine (same validation code as the lab), persists {cells, score,
// rng, moves} across hibernation, self-deletes after an hour of inactivity.
export class GameSession {
  constructor(state) {
    this.state = state;
    this.ex = null;
  }

  async engine() {
    if (!this.ex) {
      const inst = await WebAssembly.instantiate(ENGINE);
      this.ex = inst.exports;
    }
    return this.ex;
  }

  readResult(ptr) {
    const len = this.ex.result_len();
    return JSON.parse(
      new TextDecoder().decode(new Uint8Array(this.ex.memory.buffer, ptr, len))
    );
  }

  writeInput(arr) {
    const ptr = this.ex.input_ptr();
    new Uint32Array(this.ex.memory.buffer, ptr, arr.length).set(arr);
  }

  // Rehydrate the wasm game from storage (after hibernation). Returns the
  // saved dump, or null if this session doesn't exist.
  async load() {
    await this.engine();
    const saved = await this.state.storage.get("g");
    if (!saved) return null;
    this.writeInput([...saved.cells, saved.score, saved.rng, saved.moves]);
    this.ex.game_restore();
    return saved;
  }

  async persist() {
    const dump = this.readResult(this.ex.game_dump());
    await this.state.storage.put("g", dump);
    await this.state.storage.setAlarm(Date.now() + 3600_000);
    return dump;
  }

  async alarm() {
    await this.state.storage.deleteAll();
  }

  async fetch(req) {
    const path = new URL(req.url).pathname;

    if (path === "/new") {
      await this.engine();
      const seed = new Uint32Array(1);
      crypto.getRandomValues(seed);
      const st = this.readResult(this.ex.game_new(seed[0]));
      const dump = await this.persist();
      return Response.json({
        cells: st.cells, score: st.score, over: st.over, moves: dump.moves,
      });
    }

    if (path === "/state") {
      const saved = await this.load();
      if (!saved) return Response.json({ error: "no such game" }, { status: 404 });
      const st = this.readResult(this.ex.game_state());
      return Response.json({
        cells: st.cells, score: st.score, over: st.over, moves: saved.moves,
      });
    }

    if (path === "/move") {
      const saved = await this.load();
      if (!saved) return Response.json({ error: "no such game" }, { status: 404 });
      if (saved.moves >= 3000) return Response.json({ error: "move cap" }, { status: 400 });
      const body = await req.json().catch(() => ({}));
      const p = body.path;
      // sequence check: a retried request that already applied must not
      // double-apply; client resyncs from the returned state instead
      if (Number.isInteger(body.n) && body.n !== saved.moves) {
        const st0 = this.readResult(this.ex.game_state());
        return Response.json({
          error: "desync",
          cells: st0.cells, score: st0.score, over: st0.over, moves: saved.moves,
        }, { status: 409 });
      }
      if (
        !Array.isArray(p) || p.length < 2 || p.length > 25 ||
        !p.every((c) => Number.isInteger(c) && c >= 0 && c < 25)
      ) {
        return Response.json({ error: "bad path" }, { status: 400 });
      }
      this.writeInput(p);
      if (!this.ex.game_apply(p.length)) {
        return Response.json({ error: "illegal move" }, { status: 400 });
      }
      const st = this.readResult(this.ex.game_state());
      const dump = await this.persist();
      return Response.json({
        cells: st.cells, score: st.score, over: st.over, moves: dump.moves,
      });
    }

    if (path === "/finish") {
      const saved = await this.load();
      if (!saved) return Response.json({ error: "no such game" }, { status: 404 });
      const st = this.readResult(this.ex.game_state());
      if (!st.over) return Response.json({ error: "game not over" }, { status: 400 });
      await this.state.storage.deleteAll();
      return Response.json({ score: st.score, moves: saved.moves });
    }

    return Response.json({ error: "not found" }, { status: 404 });
  }
}
