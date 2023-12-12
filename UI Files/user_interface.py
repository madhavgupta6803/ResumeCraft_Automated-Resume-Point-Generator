# -*- coding: utf-8 -*-
"""User_Interface.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u_ynGZA8hoTyJ7b22lSirWDn4B639MoE
"""

!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -q bitsandbytes einops

!pip install chainlit pyngrok

!chainlit run app.py &>/content/logs.txt &

!ngrok config add-authtoken 2Yu6XJdo2F6vLQTG2B9krjlNFE2_2TX2VKyUvwfovrwaJGsSd

from pyngrok import ngrok
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)

from pyngrok import ngrok
tun = ngrok.get_tunnels()
tun

ngrok.kill()