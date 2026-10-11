```{bash}
git clone --recursive git@github.com:seanv507/xgboost.git
# git submodule update --init --recursive
cd xgboost/
git checkout cat_order_smooth
cd ..
wget https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-linux.zip
unzip ninja-linux.zip 
chmod +x ninja
cp ninja /usr/local/bin/
cd xgboost/
cmake -B build -S . -DUSE_CUDA=ON -GNinja
cd build && ninja
#cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja


cd ..
cd python-package/
uv pip install -e .
uv pip install wandb_workspaces matplotlib
echo $PATH
python -m ipykernel install --user --name=myenv
```
download ninja

```
```
