#!/usr/bin/env bash
set -euo pipefail

# 1. Enter the raw data directory
cd ./data/imagenet64_raw/

# 2. Unzip all three archives (overwrite if exists)
for zipfile in \
    Imagenet64_train_part1.zip \
    Imagenet64_train_part2.zip \
    Imagenet64_val.zip; do
  echo "Unzipping $zipfile..."
  unzip -o "$zipfile"
done

# 3. Move all extracted files up to ./data/
cd ../..   # back to project root
echo "Moving extracted batches to ./data/ ..."
mv data/imagenet64_raw/train_data_batch_* data/
mv data/imagenet64_raw/val_data       data/

# 4. Remove the now-empty raw folder (and any remaining zips)
echo "Removing raw folder..."
rm -rf data/imagenet64_raw

echo "Done."
