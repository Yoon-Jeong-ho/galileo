# SSH key setup for `nlp16`

This guide sets up **key-based SSH login** for the server:

- Host: `nlp16`
- HostName: `163.152.163.64`
- User: `aa007878`
- Port: `57022`
- Remote workdir: `/mnt/raid6/aa007878/`

> Goal: enable passwordless (or agent-backed) SSH so long-running `tmux` experiments can be started/monitored reliably.

## 1) Add host entry

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh

cat >> ~/.ssh/config <<'EOF'
Host nlp16
  HostName 163.152.163.64
  User aa007878
  Port 57022
  IdentityFile ~/.ssh/nlp16_ed25519
  IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
```

## 2) Generate a key

```bash
ssh-keygen -t ed25519 -f ~/.ssh/nlp16_ed25519 -C "openclaw@nlp16"
```

## 3) Install the public key

### Option A (preferred): `ssh-copy-id`

```bash
ssh-copy-id -i ~/.ssh/nlp16_ed25519.pub -p 57022 aa007878@163.152.163.64
```

### Option B: manual

1) Print your public key:

```bash
cat ~/.ssh/nlp16_ed25519.pub
```

2) SSH in (password) once and append it:

```bash
ssh -p 57022 aa007878@163.152.163.64
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys  # or vi
chmod 600 ~/.ssh/authorized_keys
exit
```

## 4) Test

```bash
ssh nlp16 'hostname && whoami'
```

If it does not prompt for a password, setup is complete.

## Notes

- If you set a passphrase, you may need `ssh-agent` (or `keychain`) so non-interactive runs can reconnect.
- Keep private keys restricted:

```bash
chmod 600 ~/.ssh/nlp16_ed25519
```
