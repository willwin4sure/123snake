#!/bin/zsh
# Daily export of the production scores table to local disk.
# scores only (name/score/ts/moves — already public via the API);
# the rate-limit and visit tables hold ip hashes and are not exported.
cd /Users/willyue/Documents/integer_snake/server
mkdir -p ../backups
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
STAMP=$(date +%Y%m%d)
npx -y wrangler d1 export snake-scores --remote --table scores \
  --output "../backups/scores-$STAMP.sql" >> ../backups/backup.log 2>&1 \
  && echo "$(date '+%F %T') ok scores-$STAMP.sql" >> ../backups/backup.log
# keep ~60 days
find ../backups -name 'scores-*.sql' -mtime +60 -delete
