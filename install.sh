sudo apt-get install git

git clone https://github.com/aaalgo/kgraph.git
cd kgraph
cmake -DCMAKE_BUILD_TYPE=release .
make
sudo make install
python setup.py install

pip install pybind11 cython numpy sklearn matplotlib scipy
pip install pyflann annoy hdidx n2 nmslib
