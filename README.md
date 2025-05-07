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
python upload.py
```

# Deploy
```bash
uvicorn server:app --reload
open client.html
``` 