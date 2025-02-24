
# Download EDM pretrained models
mkdir pretrained
wget -O pretrained/edm-cifar10-32x32-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
wget -O pretrained/edm-cifar10-32x32-uncond-ve.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl
wget -O pretrained/edm-ffhq-64x64-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl
wget -O pretrained/edm-afhqv2-64x64-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl

# Download LDM pretrained models
mkdir -p pretrained/ldm/cin256-v2/
wget -O pretrained/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt

mkdir -p pretrained/first_stage_models/vq-f4
wget -O pretrained/first_stage_models/vq-f4/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
cd pretrained/first_stage_models/vq-f4
unzip -o model.zip
cd ../../..

mkdir -p pretrained/ldm/lsun_beds256
wget -O pretrained/ldm/lsun_beds256/lsun_beds-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip
cd pretrained/ldm/lsun_beds256
unzip -o lsun_beds-256.zip
cd ../../..

# Stable Diffusion
mkdir -p pretrained/ldm/stable-diffusion-v1/
wget -O pretrained/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt

# Download FID references
mkdir fid-refs
wget -O fid-refs/afhqv2-64x64.npz https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz
wget -O fid-refs/ffhq-64x64.npz https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz
wget -O fid-refs/cifar10-32x32.npz https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
wget -O fid-refs/VIRTUAL_lsun_bedroom256.npz https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz
wget -O fid-refs/VIRTUAL_imagenet256_labeled.npz https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
