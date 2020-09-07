#!/bin/bash

TARGET_DIR=$1

if [ ! $# -eq 1 ]
then
	echo "Target directory must be provided (Only 1)."
	exit 1
fi

INPUT=$(find $TARGET_DIR -maxdepth 1 -type f -name '*' | nl | paste)

cat <<< $INPUT | while read -r ID FILE
do
	echo "mkdir $TARGET_DIR/$ID"
	echo "mv  $FILE $TARGET_DIR/$ID"
done
