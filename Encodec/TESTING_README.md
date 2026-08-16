Activate and deactivatie virtual environment

deactivate

venv_encodec\Scripts\activate

Check if python can find torch

python -c "import torch; print(torch.__version__)"

Basic encoding of batches with 24kbps

python .\batch_encode_24kbps.py .\Audio_Tests_AL\ 24 cuda

Quick check to ensure ecdc can be decoded correctly

python main.py .\Audio_Tests_AL\baseline_0mm_bw24.0.ecdc test_decode.wav --model_name encodec_48khz --device cuda --force

Basic tokenizing of bit streams

python ecdc_to_tokens_npy.py --input .\Audio_Tests_AL\ --model-name encodec_48khz --device cuda