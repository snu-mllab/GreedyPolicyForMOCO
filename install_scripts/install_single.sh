conda create -n setbench python=3.9 -y && conda activate setbench
pip install -r requirements.txt
pip install botorch==0.6.5 gpytorch==1.8.0 torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install protobuf==3.20
pip install -e .
pip install cvxopt
