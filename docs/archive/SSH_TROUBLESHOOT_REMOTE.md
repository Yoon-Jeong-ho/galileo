# SSH troubleshooting: remote host access (nlp8 SSOT)

This note is for restoring `ssh` access from an automated runtime.

**Current SSOT for EMNLP Main experiments:** `ssh nlp8`.

## 1) Minimal info to share (safe)

Paste either:
- your `Host nlp8` block from `~/.ssh/config` (you can remove comments and any unrelated hosts), or
- the equivalent values:
  - HostName (IP/domain)
  - User
  - IdentityFile path (key filename)
  - ProxyJump / ProxyCommand (if any)

## 2) Quick diagnostics to run locally

```bash
ssh -v nlp8 'hostname; whoami'
```

Common failure modes:
- `Permission denied (publickey)` → key not offered / wrong user / server doesn’t accept your key
- `no such identity` → IdentityFile path incorrect or key not present
- `Too many authentication failures` → SSH agent has many keys; add `IdentitiesOnly yes`

## 3) Fix patterns (typical)

### A) Ensure we use the intended key

In `~/.ssh/config`:

```sshconfig
Host nlp8
  HostName <host>
  User <user>
  IdentityFile ~/.ssh/<keyname>
  IdentitiesOnly yes
```

### B) Ensure the key is loaded (if using ssh-agent)

```bash
ssh-add -l
ssh-add ~/.ssh/<keyname>
```

### C) Make sure permissions are correct

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/<keyname>
chmod 644 ~/.ssh/<keyname>.pub
```

## 4) What I need to proceed

- Confirm whether `ssh nlp8` works from *your* terminal.
- If yes, paste the `Host nlp8` stanza (sanitized is fine) and tell me which key file to use.
