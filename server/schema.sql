-- 123 Snake leaderboard schema (D1 / SQLite)

CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,          -- ms since epoch (UTC)
  ini TEXT NOT NULL,            -- three uppercase letters
  score INTEGER NOT NULL,
  moves INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS scores_ts_score ON scores(ts, score DESC);

-- per-IP-hash game-creation log for rate limiting (pruned opportunistically)
CREATE TABLE IF NOT EXISTS newlog (
  ts INTEGER NOT NULL,
  iph TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS newlog_iph_ts ON newlog(iph, ts);

-- visit counter: daily uniques by hashed ip+day, plus an all-time total
CREATE TABLE IF NOT EXISTS hits (
  day TEXT NOT NULL,
  iph TEXT NOT NULL,
  PRIMARY KEY (day, iph)
);
CREATE TABLE IF NOT EXISTS meta (
  k TEXT PRIMARY KEY,
  v INTEGER NOT NULL
);
