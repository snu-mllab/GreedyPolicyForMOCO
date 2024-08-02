conda create -n setbench python=3.8 -y && conda activate setbench
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install protobuf==3.20
pip install -e .
pip install cvxopt