if [ -z $STEREO_SOURCE_DIR ]; then echo 'Please source "setup_env.sh" first'
else
wget -O ndf_obj_assets.tar.gz https://www.dropbox.com/s/831szjnb8l7gbdh/ndf_obj_assets.tar.gz?dl=0
mv ndf_obj_assets.tar.gz $STEREO_SOURCE_DIR/data
cd $STEREO_SOURCE_DIR/data
tar -xzf ndf_obj_assets.tar.gz
rm ndf_obj_assets.tar.gz
echo "Object models for NDF copied to $STEREO_SOURCE_DIR/data"

cd $STEREO_SOURCE_DIR
wget -O ndf_other_assets.tar.gz https://www.dropbox.com/s/fopyjjm3fpc3k7i/ndf_other_assets.tar.gz?dl=0
mkdir $STEREO_SOURCE_DIR/data/assets
mv ndf_other_assets.tar.gz $STEREO_SOURCE_DIR/data/assets
cd $STEREO_SOURCE_DIR/data/assets
tar -xzf ndf_other_assets.tar.gz
rm ndf_other_assets.tar.gz
echo "Additional object-related assets copied to $STEREO_SOURCE_DIR/data/assets"
fi