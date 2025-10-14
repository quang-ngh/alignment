pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install omegaconf

echo "Installing HPSv2"
cd HPSv2
pip install -r requirements.txt
pip install -e .
python setup.py develop


echo "Install ImageReward"
cd ../ImageReward
python setup.py develop
cd ..

echo "PickScore"
pip install fire
cd PickScore
python setup.py develop
cd ..

if [ ! -d "checkpoints" ]; then
  mkdir checkpoints
fi

python download_hf.py repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" local_dir=checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K
python download_hf.py repo_id="yuvalkirstain/PickScore_v1" local_dir=checkpoints/pickscore_v1
