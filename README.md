
运行方式：
首先，通过git clone将文件复制到您的本地文件夹
然后，在2048-api/game2048/train.py中修改'sys.path.append'中的路径为您当前game2048文件夹所处的绝对路径，之后可以开始运行

运行环境：
Python 3
PyTorch >= 0.4.0
numpy

训练：首先进入game2048文件夹

cd game2048
python train.py

模型即开始自动进行online训练

若通过远程服务器进行训练，可以通过提供的run.sh文件

sbatch run.sh

运行时间可以在文件内修改

测试：可以直接通过evaluate.py进行测试，或者通过webapp.py观察结果，测试与训练是完全分离的。
python evaluate.py >> 1.log
