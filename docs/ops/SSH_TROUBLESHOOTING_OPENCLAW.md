# SSH Troubleshooting (OpenClaw / WSL)

This note is for the common failure:

- `Permission denied (publickey,password)`

when attempting to run remote GALILEO experiments (e.g., `ssh nlp8`, `ssh nlp16`) from the machine that runs OpenClaw.

## 0) Quick checklist (fastest path)

On the OpenClaw host/WSL:

```bash
ssh -v nlp8 'hostname'
# or
ssh -v nlp16 'hostname'
```

Then check:

1. **Which key is being offered?**
   - In `-v` logs, look for `Offering public key:`
2. **Is the correct IdentityFile *and Port* configured?**
   - `ssh -G nlp8  | egrep -i 'identityfile|port|hostname|user'`
   - `ssh -G nlp16 | egrep -i 'identityfile|port|hostname|user'`
   - If `port 22` shows up for `nlp16`, your `~/.ssh/config` entry is not being applied in *this* environment.
3. **Does the key file exist and have correct perms?**
   - `ls -l ~/.ssh/*.pub ~/.ssh/*ed25519 2>/dev/null`
   - `chmod 700 ~/.ssh && chmod 600 ~/.ssh/config ~/.ssh/<key>`
4. **Is an ssh-agent needed?** (only if key has a passphrase)
   - `ssh-add -l` (should list your key)

If the key is *not* being offered, fix `~/.ssh/config` first.

If the key **is** offered but auth still fails, run `ssh -vvv nlp16 'hostname'` and look for patterns:

- If you see:
  - `Offering public key: ...`
  - `Server accepts key: ...`
  - then `we did not send a packet, disable method`

  This usually means the private key is **encrypted (passphrase)** but you're running in non-interactive/BatchMode context, so SSH cannot prompt to sign. Fix by loading the key into an agent:

  ```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/nlp16_ed25519
  ssh-add -l
  ```

  After that, `ssh -o BatchMode=yes nlp16 'hostname'` should work.

## 1) Minimal known-good config templates

### nlp8

```sshconfig
Host nlp8
  HostName 163.152.163.182
  User aa007878
  Port 48022
  IdentityFile ~/.ssh/nlp8_openclaw_ed25519
  IdentitiesOnly yes
```

### nlp16

```sshconfig
Host nlp16
  HostName 163.152.163.64
  User aa007878
  Port 57022
  IdentityFile ~/.ssh/nlp16_ed25519
  IdentitiesOnly yes
```

Then:

```bash
chmod 600 ~/.ssh/config
```

## 2) Confirm the public key is installed server-side

On the server (requires one-time login via password or existing working key):

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Ensure the `ssh-ed25519 ...` line is on **one line** (no wrapping).

## 3) Debug commands that usually pinpoint the issue

```bash
# Show resolved SSH config
ssh -G nlp8 | sed -n '1,120p'

# Force only the intended key
ssh -i ~/.ssh/nlp8_openclaw_ed25519 -o IdentitiesOnly=yes -p 48022 aa007878@163.152.163.182 'hostname'

# Show what keys are loaded in agent
ssh-add -l
```

## 4) OpenClaw context note

If OpenClaw runs inside a different environment (WSL, container, different user),
make sure the **same** `~/.ssh/` (keys + config) exists *in that environment*.

- Common pitfall: keys exist on Windows host but not inside WSL.

## 5) Once SSH works: experiment conventions

- Always set `CUDA_VISIBLE_DEVICES` according to the **current experiment policy** (see the repo/workspace `HEARTBEAT.md`, which we sometimes update temporarily due to contention).
- Use `tmux` to keep runs alive.
- Keep worker counts small to avoid CPU overload.
