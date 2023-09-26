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
    echo "##########################################################"
    echo "##########################################################" 
    echo "$file has errors:"
    echo "args:"
    file2="${file%.err}.out"
    head -4 "$file2" | sed 's/^/\t/'
    echo "" 
    echo "error:" 
    echo "" 
    head -100 "$file" | sed 's/^/\t/'
  else
    echo "##########################################################"
    echo "$file does not have errors"
  fi
done < <(find "$ROOT" -name 'log.*.err' -print0)

exit 0
