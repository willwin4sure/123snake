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

// Serious-only blocklist for 3-letter initials (exact matches).
const BAD_INITIALS = new Set([
  "FAG", "NIG", "NGR", "KKK", "JEW", "GAS", "FUK", "FUC", "FCK", "CUM",
  "SEX", "ASS", "DIE", "KYS", "NAZ", "RAP",
]);

async function sha1hex(s) {
  const d = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
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
      const iph = (await sha1hex(ip)).slice(0, 16);
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
      const { id, path } = body;
      if (typeof id !== "string" || !/^[0-9a-f]{32}$/.test(id) || !Array.isArray(path)) {
        return json({ error: "bad request" }, 400);
      }
      const stub = env.GAME.get(env.GAME.idFromName(id));
      const r = await stub.fetch("https://do/move", {
        method: "POST",
        body: JSON.stringify({ path }),
      });
      return json(await r.json(), r.status);
    }

    if (p === "/api/finish" && req.method === "POST") {
      const body = await req.json().catch(() => ({}));
      const { id, ini } = body;
      if (typeof id !== "string" || !/^[0-9a-f]{32}$/.test(id)) {
        return json({ error: "bad request" }, 400);
      }
      const clean = String(ini || "").toUpperCase().replace(/[^A-Z]/g, "").slice(0, 3);
      if (clean.length !== 3) return json({ error: "initials must be 3 letters" }, 400);
      if (BAD_INITIALS.has(clean)) return json({ error: "pick different initials" }, 400);
      const stub = env.GAME.get(env.GAME.idFromName(id));
      const r = await stub.fetch("https://do/finish", { method: "POST" });
      if (!r.ok) return json(await r.json(), r.status);
      const fin = await r.json();
      await env.DB.prepare(
        "INSERT INTO scores(ts, ini, score, moves) VALUES(?,?,?,?)"
      ).bind(Date.now(), clean, fin.score, fin.moves).run();
      const better = await env.DB.prepare(
        "SELECT COUNT(*) AS n FROM scores WHERE score>? AND ts>=?"
      ).bind(fin.score, Date.UTC(new Date().getUTCFullYear(), new Date().getUTCMonth(), new Date().getUTCDate())).first();
      return json({ ok: 1, score: fin.score, daily_rank: better.n + 1 });
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
      const { results } = await env.DB.prepare(
        "SELECT ini, score, ts FROM scores WHERE ts>=? ORDER BY score DESC, ts ASC LIMIT 25"
      ).bind(since).all();
      return json({ rows: results });
    }

    if (p === "/api/hit" && req.method === "POST") {
      const ip = req.headers.get("cf-connecting-ip") || "0";
      const day = new Date().toISOString().slice(0, 10);
      const iph = (await sha1hex(ip + day)).slice(0, 16);
      await env.DB.prepare("INSERT OR IGNORE INTO hits(day, iph) VALUES(?,?)")
        .bind(day, iph).run();
      await env.DB.prepare(
        "INSERT INTO meta(k, v) VALUES('total', 1) ON CONFLICT(k) DO UPDATE SET v=v+1"
      ).run();
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
      await this.persist();
      return Response.json({ cells: st.cells, score: st.score, over: st.over });
    }

    if (path === "/move") {
      const saved = await this.load();
      if (!saved) return Response.json({ error: "no such game" }, { status: 404 });
      if (saved.moves >= 3000) return Response.json({ error: "move cap" }, { status: 400 });
      const body = await req.json().catch(() => ({}));
      const p = body.path;
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
      await this.persist();
      return Response.json({ cells: st.cells, score: st.score, over: st.over });
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
