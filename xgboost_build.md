```{bash}
git clone --recursive git@github.com:seanv507/xgboost.git
# git submodule update --init --recursive
cd xgboost/
git checkout cat_order_smooth

cd python-package/
uv pip install -e .
echo $PATH
```
download ninja

```
cd ..
wget https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-linux.zip
unzip ninja-linux.zip 
chmod +x ninja
cp ninja /usr/local/bin/
cd xgboost/
cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cmake -B build -S . -DUSE_CUDA=ON -GNinja
cd build && ninja
```
