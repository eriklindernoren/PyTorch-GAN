FILE=$dataset-v1
TAR_FILE=./$FILE.zip
URL="https://drive.google.com/uc?id=1Yt_jVhj-eBaGsb8fXhECcoKZtA9T70q9&export=download"
gdown $URL -O TAR_FILE
unzip TAR_FILE
rm -r TAR_FILE