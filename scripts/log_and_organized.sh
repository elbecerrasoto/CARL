#!/bin/bash

OUT_DIR='runs'
TARGET_FILES=$OUT_DIR/target_csv.txt

mkdir -p $OUT_DIR
find ./ -maxdepth 2 -type f -name '*.csv' > $TARGET_FILES

cat $TARGET_FILES | while read -r FILE
do
	CURRENT_OUT=$OUT_DIR/$(dirname $FILE)
	logs2tb $FILE --outdir $CURRENT_OUT
done
