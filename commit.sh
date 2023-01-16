if [ $# -eq 0 ]
then
    message="updated";
else
    message=$1;
fi
echo "commit with message: $message";
git add -A --verbose
git commit -a -m "$message"
git push --all