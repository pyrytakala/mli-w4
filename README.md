# Run tests
`python test_data.py`

# Train
Start machine and update its IP https://computa.mlx.institute/
`open ~/.ssh/config`
`scp -C * computa:~/`

```bash
ssh computa
apt update && apt install -y tmux vim
pip install -r requirements.txt
tmux
python train.py
```

# Deploy
```bash
scp -C computa:~/checkpoints/epoch_20.pth checkpoints/.
python server.py
open client.html
``` 