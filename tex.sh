#!/bin/sh

output=all_files.tex
rm $output
git branch -a | grep remotes | sed 's/_2/_,2/' | awk -F, '{print $2,$1}' | sort | awk '{print $2$1}' | head  -n-3 | while read branch; do
    echo "# branch: $branch" >> $output
    echo '```tex' >> $output
    git show ${branch}:will_handley.tex >> $output
    git show ${branch}:talk.tex >> $output
    echo '```' >> $output
done
