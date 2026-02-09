# NLP8 SSH Key Setup (OpenClaw / non-interactive)

This note mirrors the NLP16 setup doc, but for **nlp8**.

- Host: `nlp8`
- HostName: `163.152.163.182`
- User: `aa007878`
- Port: `48022`
- GPU policy (GALILEO convention): use **GPUs 4,5,6,7**.

## 1) Create a dedicated keypair (recommended)

On the machine where OpenClaw runs (or your local WSL if you also SSH from there):

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh

ssh-keygen -t ed25519 \
  -f ~/.ssh/nlp8_openclaw_ed25519 \
  -C openclaw@nlp8 \
  -N ""

ls -l ~/.ssh/nlp8_openclaw_ed25519 ~/.ssh/nlp8_openclaw_ed25519.pub
```

## 2) Register the public key on nlp8

You need *one* initial login to nlp8 (password or existing key) to append the pubkey.

On your local machine:

```bash
cat ~/.ssh/nlp8_openclaw_ed25519.pub
```

Copy the single `ssh-ed25519 ... openclaw@nlp8` line.

On **nlp8** (after you log in):

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Append the copied public key line
nano ~/.ssh/authorized_keys

chmod 600 ~/.ssh/authorized_keys
```

(Keep the public key on a **single line**; do not wrap.)

## 3) Add a host alias in `~/.ssh/config`

On the client machine (where you run `ssh nlp8`):

```bash
nano ~/.ssh/config
```

Append:

```sshconfig
Host nlp8
  HostName 163.152.163.182
  User aa007878
  Port 48022
  IdentityFile ~/.ssh/nlp8_openclaw_ed25519
  IdentitiesOnly yes
```

Then:

```bash
chmod 600 ~/.ssh/config
```

## 4) Test

```bash
ssh -v nlp8 'hostname'
ssh nlp8 'nvidia-smi -L'
```

If it still asks for a password, re-check `authorized_keys` and that the `IdentityFile` path exists.

## 5) Experiment runner notes (nlp8)

- Prefer splitting runs across GPUs rather than TP=4 for everything.
- Example: run control vs persona on separate GPUs:

```bash
# control on GPU4
CUDA_VISIBLE_DEVICES=4 \
  conda run -n galileo python run_experiment.py ... --tensor_parallel_size 1 --personas control_reask

# persona on GPU5
CUDA_VISIBLE_DEVICES=5 \
  conda run -n galileo python run_experiment.py ... --tensor_parallel_size 1 --personas no_control
```

Use `tmux` so runs survive disconnects.
