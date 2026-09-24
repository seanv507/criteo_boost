```{bash}
git clone --recursive git@github.com:seanv507/xgboost.git
# git submodule update --init --recursive
cd python-package/
uv pip install -e .
echo $PATH
```
download ninja

```
unzip ninja-linux.zip 
chmod +x ninja
cp ninja /usr/local/bin/
cd xgboost/
cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cd build && ninja
```
