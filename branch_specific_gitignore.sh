#!/bin/sh

branch=$1
root_dir="$(pwd -P)"
info_dir="$root_dir/.git/info"

exclude_target=".gitignore"
if [ -f "$root_dir/$exclude_target.$branch" ]
then
    echo "Prepare to use .gitignore.$branch as exclude file"
    exclude_target=.gitignore.$branch
else
    echo "Usbe the default .gitignore as exclude file"
fi

cd "$info_dir"
rm exclude
echo "Copy $exclude_target file in place of exclude"
cp "$root_dir/$exclude_target" exclude