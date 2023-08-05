#!/bin/bash
# example use: $ ./report_slurm_errors /home/exacloud/gscratch/NGSdev/evans/GSNN/output/

ROOT=$1
TAB=$(echo -e "\t")

# Check if ROOT directory exists
if [ ! -d "$ROOT" ]; then
  echo "Error: Directory $ROOT does not exist."
  exit 1
fi

# Iterate over each .err file in the directory structure
while IFS= read -r -d '' file
do
  # Check if the file has any content
  if [ -s "$file" ]
  then
    echo "$file has errors. The first 10 lines of errors are:"
    head -100 "$file" | sed 's/^/\t/'
  else
    echo "$file does not have errors"
  fi
done < <(find "$ROOT" -name 'log.*.err' -print0)

exit 0
