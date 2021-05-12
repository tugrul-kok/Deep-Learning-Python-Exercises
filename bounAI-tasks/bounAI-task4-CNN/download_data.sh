if [ ! -f data/ ]
then

#This will download data from https://drive.google.com/drive/folders/1_xOtuwg8jmmoQ4fK_PUBj5c8MCttEnUB?usp=sharing

pip3 install gdown
gdown https://drive.google.com/u/0/uc?id=1_xOtuwg8jmmoQ4fK_PUBj5c8MCttEnUB?usp=sharing

else
  echo "data exists"
fi
