if [ ! -f data/ ]
then

mkdir data
cd data
wget -cq http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar -xvf annotation.tar
rm annotation.tar

else
  echo "data exists"
fi
