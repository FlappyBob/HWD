# init conda env
conda create --name hwd
conda activate hwd
python setup.py sdist bdist_wheel
pip install .
pip install -r requirements.txt

# get model card from huggingface
git clone https://huggingface.co/Ruian7P/emuru_result