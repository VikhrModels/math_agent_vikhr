#!/usr/bin/env bash
set -euo pipefail

# cd to repo root
cd "$(dirname "$0")"

# Find a free port in 5000-5005
PORTS=(5000 5001 5002 5003 5004 5005)
FREE_PORT=""
for p in "${PORTS[@]}"; do
  if ! ss -ltnp | grep -q ":$p"; then
    FREE_PORT="$p"
    break
  fi
done
if [[ -z "$FREE_PORT" ]]; then
  echo "No free ports in 5000-5005." >&2
  exit 1
fi

# Activate venv
if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
else
  echo "venv not found. Please create/activate venv first." >&2
  exit 1
fi

# Ensure secrets exist
mkdir -p tmp
SECRETS_FILE="tmp/phoenix_secrets.env"
if [[ ! -f "$SECRETS_FILE" ]]; then
  python - <<'PY' > "$SECRETS_FILE"
import secrets

def strong(n):
    return secrets.token_urlsafe(n)

print("PHOENIX_SECRET=" + strong(48))
print("PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD=" + strong(16))
PY
fi

# Export auth and server env
set -a
source "$SECRETS_FILE"
set +a

export PHOENIX_ENABLE_AUTH=true
export PHOENIX_HOST=0.0.0.0
export PHOENIX_PORT="$FREE_PORT"
# Persist chosen port for other processes (agent auto-detect)
echo -n "$FREE_PORT" > tmp/phoenix_port

echo "Starting Phoenix on 0.0.0.0:${FREE_PORT} (auth enabled)."
echo "Admin initial password is set via PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD in $SECRETS_FILE"
echo "To change, edit the file and restart."

# Start Phoenix server
set +e
echo "Starting Phoenix server on port $FREE_PORT..."

# Set environment variables for Phoenix
export PHOENIX_HOST=0.0.0.0
export PHOENIX_PORT="$FREE_PORT"

# Start Phoenix using Python API
python -c "
import phoenix as px
import time

try:
    print('Starting Phoenix...')
    session = px.launch_app()
    print(f'Phoenix started on http://0.0.0.0:$FREE_PORT')
    print('Press Ctrl+C to stop...')
    
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print('Phoenix stopped.')
except Exception as e:
    print(f'Error: {e}')
"


